import os
import os.path as osp
import pickle as rick
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataLoader

from torch_geometric.utils import degree
from torch.nn import ReLU

import optuna
from optuna.trial import TrialState

import pandas
import numpy as np
from sklearn.model_selection import StratifiedKFold

import models
import main as main_script
import data_prep


def objective(trial, device, train_val_dataset):
    lr_start_from = 1e-7
    lr_start_to = 1e-2

    weight_decay_from = 1e-6
    weight_decay_to = 1

    dropout_convs_from = 0
    dropout_convs_to = 0.7

    dropout_linear_1_from = 0
    dropout_linear_1_to = 0.7

    dropout_linears_rest_from = 0
    dropout_linears_rest_to = 0.7

    positive_class_weight_from = 0.1
    positive_class_weight_to = 0.9

    batch_size = 10
    epochs = 101

    # Suggest hyper-parameters for trial:
    lr = trial.suggest_loguniform("learning_rate", lr_start_from, lr_start_to)
    weight_decay = trial.suggest_loguniform("weight_decay", weight_decay_from, weight_decay_to)
    dropout_conv_1_2 = trial.suggest_float("dropout_conv", dropout_convs_from, dropout_convs_to)
    dropout_conv_rest = dropout_conv_1_2
    dropout_lin_1 = trial.suggest_float("dropout_lin_1", dropout_linear_1_from, dropout_linear_1_to)
    dropout_lin_rest = trial.suggest_float("dropout_lin_rest", dropout_linears_rest_from, dropout_linears_rest_to)
    positive_class_weights = trial.suggest_float("positive_class_weight", positive_class_weight_from,
                                                 positive_class_weight_to)

    activation_funct = ReLU()

    len_train_val = round(len(train_val_dataset))
    half_val_size = round(len_train_val * 0.125)

    sum_max_aurpcs = 0
    sum_specificities = 0

    # stratified 4-fold cross validation using a sampling function from sklearn:
    label_array = np.zeros(shape=(2, len_train_val))
    for i, graph in enumerate(train_val_dataset):
        label_array[0][i] = i
        if graph.y == 0:
            label_array[1][i] = 0
        elif graph.y == 1:
            label_array[1][i] = 1

    num_neg_instances = np.count_nonzero(a=label_array[1])
    num_pos_instances = len(label_array[1]) - num_neg_instances
    print(f"Number of positive instances: {num_neg_instances}")
    print(f"Number of negative instances: {num_pos_instances}")

    skf = StratifiedKFold(n_splits=4)
    k = 0
    for train_index, val_index in skf.split(label_array[0], label_array[1]):
        train_dataset = train_val_dataset[train_index]
        validation_dataset = train_val_dataset[val_index]
        print(len(train_dataset), len(validation_dataset))

    ## 4-fold cross validation for each trial (75% / 25%):
    #for k in range(4):
    #    if k == 0:
    #        train_dataset = train_val_dataset[half_val_size * 2:]
    #        validation_dataset = train_val_dataset[:half_val_size * 2]
    #        # print(len(train_dataset), len(validation_dataset))
    #    elif k == 1:
    #        train_dataset = train_val_dataset[:half_val_size * 2] + train_val_dataset[half_val_size * 4:]
    #        validation_dataset = train_val_dataset[half_val_size * 2:half_val_size * 4]
    #        # print(len(train_dataset), len(validation_dataset))
    #    elif k == 2:
    #        train_dataset = train_val_dataset[:half_val_size * 4] + train_val_dataset[half_val_size * 6:]
    #        validation_dataset = train_val_dataset[half_val_size * 4:half_val_size * 6]
    #        # print(len(train_dataset), len(validation_dataset))
    #    elif k == 3:
    #        train_dataset = train_val_dataset[:len_train_val - (half_val_size * 2)]
    #        validation_dataset = train_val_dataset[len_train_val - (half_val_size * 2):]
    #        # print(len(train_dataset), len(validation_dataset))

        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        pos_intances = 0
        neg_instances = 0
        for data in train_dataset:
            if int(data.y) == 0:
                neg_instances = neg_instances + 1
            elif int(data.y) == 1:
                pos_intances = pos_intances + 1

        # build batches:
        try:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size, shuffle=True, drop_last=True)
            validation_loader = DataLoader(validation_dataset,
                                           batch_size=batch_size, shuffle=False, drop_last=True)
        except:
            print("Problem with train_loader or validation_loader. Probably batch size to large.")
            raise optuna.exceptions.TrialPruned()

        model = models.PNAnet4Lb(dropout_lin_1, dropout_lin_rest, deg, train_val_dataset.num_node_features,
                                 activation_funct)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # for CrossEntropyLoss:

        weights = torch.tensor([1, positive_class_weights]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        max_aurpc = 0
        specificity = 0
        #try:
        for epoch in range(epochs):
            if epoch % 40 == 0:
                print(f"Epoch {epoch}")
            lmean = main_script.train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device,
                                      optimizer)
            val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = main_script.test(validation_loader, 0, 0,
                                                                                       model, device)
            # for multi objective:
            if val_aurpc > max_aurpc:
                max_aurpc = val_aurpc
                # Get the specificity from the same epoch, as the best aurpc, because both should be good at the
                # same time
                specificity = val_spec
            # max_aurpc = max([max_aurpc, val_aurpc])
            if epoch % 40 == 0:
                print(f"mean loss: {lmean} \n"
                      f"Validation: \n"
                      f"acc: {val_acc}, prec: {val_prec}, rec: {val_rec}, spec: {val_spec}, f1: {val_f1}, "
                      f"aurpc: {val_aurpc}, max aurpc: {max_aurpc}")

            if epoch == 80 and k == 0:
                cross_val_1_max_aurpc = max_aurpc  # Keep track of the max value at epoch 80 of the first k.
            if epoch == 80 and k == 1:
                if cross_val_1_max_aurpc <= 0.2 and max_f1 <= 0.2:
                    print(f"Trial pruned because max F1 below 0.1 for the first 80 epochs of 1st and 2nd CV. \n"
                          f"Parameters: lr: {lr}, weight decay: {weight_decay}, dropout conv 1/2: {dropout_conv_1_2},"
                          f"dropout conv rest: {dropout_conv_rest}, dropout lin 1: {dropout_lin_1}, "
                          f"dropout lin rest: {dropout_lin_rest}, positive class weight: {positive_class_weights}")
                    raise optuna.exceptions.TrialPruned()
        # trial.report(max_aurpc, k)  # apparently, trial.report() is not implemented for multi-objective optimization.
        sum_max_aurpcs = sum_max_aurpcs + max_aurpc
        sum_specificities = sum_specificities + specificity
        k = k + 1

        #except RuntimeError as e:
        #    print("Trial failed because of RuntimeError")
        #    print(f"Hyperparameters: \n"
        #          f"Parameters: lr: {lr}, weight decay: {weight_decay}, dropout conv 1/2: {dropout_conv_1_2},"
        #          f"dropout conv rest: {dropout_conv_rest}, dropout lin 1: {dropout_lin_1}, "
        #          f"dropout lin rest: {dropout_lin_rest}, positive class weight: {positive_class_weights}"
        #          )
        #    print("cuda memory allocated:", torch.cuda.memory_allocated())
        #    print("cuda memory reserved:", torch.cuda.memory_reserved())
        #    del model
        #    del train_loader
        #    del validation_loader
        #    print("cuda memory allocated after:", torch.cuda.memory_allocated())
        #    print("cuda memory reserved after:", torch.cuda.memory_reserved())
        #    raise optuna.exceptions.TrialPruned()

    mean_max_aurpc = sum_max_aurpcs / 4
    mean_specificity = sum_specificities / 4
    return mean_max_aurpc, mean_specificity


def main(cuda_nr=0):
    study_name = "pnaconv_ril_seq_multi_obj_strat_25_08_22_test2"

    n_trials = 10

    rooot = "/data/test_ril_seq_18_08_22/"
    input_list_size = 2000

    # create a sqlite-file to store the study or load it, if it already exists:
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"], storage=storage_name,
                                load_if_exists=True)

    device = torch.device(("cuda:" + str(cuda_nr)) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    path_to_here = os.getcwd()
    path_to_raw = path_to_here + rooot + "raw/"
    path_to_processed = path_to_here + rooot + "processed/"

    content_raw = os.listdir(path_to_raw)
    dataset = data_prep.MyDataset(root="." + rooot, input_list_size=input_list_size, content_raw=content_raw,
                                  path_to_processed=path_to_processed)

    dataset = dataset.shuffle()

    max_graph_size = 400

    # Throw out all graphs that are larger than the given size. Graphs larger than 550 n are a problem with
    # the GPU memory size of wobbuffet. When using an even smaller max. graph size, the training is much faster.
    # This increase of speed is necessary, to perform enough trials in a reasonable amount of time.
    idx_smaller_graphs = []
    for i, graph in enumerate(dataset):
        if len(graph.x) <= max_graph_size:
            idx_smaller_graphs.append(i)

    dataset = dataset[idx_smaller_graphs]

    dataset = dataset[:400]  # debugging

    # The train dataset is larger here, because it is split into train and val dataset in the k-fold CV:
    train_dataset, validation_dataset, test_dataset = main_script.stratified_train_val_test_split(dataset,
                                                                                                  train_proportion=0.85,
                                                                                                  val_proportion=0.0,
                                                                                                  test_proportion=0.15)

    train_and_val_dataset = train_dataset
    print(f'Number of training and validation graphs: {len(train_and_val_dataset)}')
    #print(f'Number of validation graphs: {len(validation_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    # function to help pass arguments to objective function:
    helping_func = lambda trial: objective(trial, device, train_and_val_dataset)

    study.optimize(helping_func, n_trials=n_trials)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    #print("Best trial:")
    #trial = study.best_trials
    #print("  Value: ", trial.value)
    #print("  Params: ")
    #for key, value in trial.params.items():
    #    print("    {}: {}".format(key, value))

    print(f"done (this study can be continued: {storage_name}")


if __name__ == "__main__":
    main()
