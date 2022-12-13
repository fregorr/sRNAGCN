import os
import os.path as osp
import pickle as rick
import statistics
import torch
from torch.nn import ReLU, LeakyReLU
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from tqdm import tqdm


# Helper scripts:
import data_prep
import data_prep_b
import models
import utils


def train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer, more_modified,
          binary_classification):
    model.train()
    batch_nr = 0
    losses = 0
    for batch in tqdm(train_loader):
        batch_nr = batch_nr + 1
        batch.to(device)
        optimizer.zero_grad()
        if more_modified:
            out = model(batch, dropout_conv_1_2, dropout_conv_rest, device)
        else:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch, batch.covalent_edges,
                        dropout_conv_1_2, dropout_conv_rest)
        if binary_classification:
            pred = out.squeeze()
            target = batch.y.float()
        else:
            pred = out
            target = batch.y
        #print(f"out: {pred}, \ntarget: {target}")
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        losss = loss.detach()
        losses = losses + losss.item()
    mean_loss = losses / batch_nr
    return mean_loss


def calculate_mae(x, y):
    sum_of_errors = 0
    for i, j in enumerate(y):
        err = x[i] - j
        sum_of_errors = sum_of_errors + abs(err)
    mae = sum_of_errors / len(y)
    return mae


def metrics(preds, y):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i in range(len(preds)):
        if preds[i] == 1:
            if preds[i] == y[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
        elif preds[i] == 0:
            if preds[i] == y[i]:
                true_negatives = true_negatives + 1
            else:
                false_negatives = false_negatives + 1
    try:
        accuracy = (true_positives + true_negatives) / len(preds)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        f1 = 2 * ((precision * recall) / (precision + recall))
    except:
        accuracy, precision, recall, specificity, f1 = 0, 0, 0, 0, 0
    #print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    return accuracy, precision, recall, specificity, f1


def test(loader, dropout_conv_1_2, dropout_conv_rest, model, device, more_modified, binary_classification):
    model.eval()
    test_predictions = []
    proba_positiv_prediction = []
    test_labels = []
    batches = 0
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for batch in tqdm(loader):
            batch.to(device)
            if more_modified:
                out = model(batch, dropout_conv_1_2, dropout_conv_rest, device)
            else:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch,
                            batch.covalent_edges, dropout_conv_1_2, dropout_conv_rest)
            pred = out.detach()
            if binary_classification:
                pred = pred >= 0
                pred = pred.long()
            else:
                pred = pred.argmax(dim=1)
                pred = pred.squeeze()
            pred = pred.tolist()
            t_labels = batch.y.detach()
            t_labels = t_labels.tolist()
            #print(f'Pred: {pred}, \n   Y: {t_labels}')
            test_predictions = test_predictions + pred
            test_labels = test_labels + t_labels
            batches = batches + 1
            # additional area under the precision recall curve:
            if binary_classification:
                proba_ones = out.detach().cpu().tolist()
            else:
                pred_probas = out.softmax(dim=1).detach().cpu()
                proba_ones = pred_probas[:, 1].tolist()  # Only the probability of class one is needed for the AUPRC (softmax
                                                # probabilities add up to 100 % so no information is lost...)
            proba_positiv_prediction = proba_positiv_prediction + proba_ones
        # mae = calculate_mae(test_predictions, test_labels)
    acc, prec, rec, spec, f1 = metrics(test_predictions, test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels, proba_positiv_prediction)
    area_under_rpc_optimistic = auc(recall, precision)  # Due to interpolating between points, this function can be
    # overly optimistic. For some cases, it returns good values for bad predictions, which is a big problem when using
    # them for HPO.
    area_under_rpc = average_precision_score(test_labels, proba_positiv_prediction)  # Summarizes an RP-curve as well
    # and is more honest...
    return acc, prec, rec, spec, f1, area_under_rpc


def load_trained_model(file_name_model_state, device, num_node_features, model_name=models.PNAnet4Lb):
    checkpoint_dict = torch.load(file_name_model_state)
    if "activation_funct" in checkpoint_dict:
        activation_funct = checkpoint_dict["activation_funct"]
    else:
        activation_funct = ReLU()
    deg = checkpoint_dict["deg"]
    if "dropout_lin_1" in checkpoint_dict:
        dropout_lin_1 = checkpoint_dict["dropout_lin_1"]
    else:
        dropout_lin_1 = 0.5
    if "dropout_lin_rest" in checkpoint_dict:
        dropout_lin_rest = checkpoint_dict["dropout_lin_rest"]
    else:
        dropout_lin_rest = 0.5
    epoch = checkpoint_dict["epoch"]
    model = model_name(dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.to(device)
    return model


def stratified_train_val_test_split(dataset, train_proportion, val_proportion, test_proportion):
    positive_idxs = [i for i, x in enumerate(dataset) if x.y == 1]
    negative_idxs = [i for i, x in enumerate(dataset) if x.y == 0]
    positive_idxs_train = positive_idxs[:round(len(positive_idxs) * train_proportion)]
    negative_idxs_train = negative_idxs[:round(len(negative_idxs) * train_proportion)]
    positive_idxs_val = positive_idxs[round(len(positive_idxs) * train_proportion):
                                      (round(len(positive_idxs) * train_proportion) +
                                       round(len(positive_idxs) * val_proportion))]
    negative_idxs_val = negative_idxs[round(len(negative_idxs) * train_proportion):
                                      (round(len(negative_idxs) * train_proportion) +
                                      round(len(negative_idxs) * val_proportion))]
    positive_idxs_test = positive_idxs[(round(len(positive_idxs) * train_proportion) +
                                       round(len(positive_idxs) * val_proportion)):]
    negative_idxs_test = negative_idxs[(round(len(negative_idxs) * train_proportion) +
                                       round(len(negative_idxs) * val_proportion)):]
    idxs_train = positive_idxs_train + negative_idxs_train
    idxs_val = positive_idxs_val + negative_idxs_val
    idxs_test = positive_idxs_test + negative_idxs_test
    train_dataset = dataset[idxs_train]
    validation_dataset = dataset[idxs_val]
    test_dataset = dataset[idxs_test]
    return train_dataset, validation_dataset, test_dataset


def train_model_with_intermediate_saving(minimum_auprc, number_of_runs, gpu_nr=0, more_modified=False,
                                         binary_classification=False, upsampling=True):
    #file_name_base = "ril_seq_train_for_benchmark_model_b"
    file_name_base = "ril_seq_run_a2_12"
    date = "28_10_22"

    epochs = 51
    batch_size = 10

    max_graph_size = 700

    activation_funct = ReLU()

    lr = 0.00005598
    weight_decay = 0.00001471

    dropout_conv_1_2 = 0.033686
    ##
    dropout_conv_rest = 0.001152  # not used, just a relict...
    ##

    dropout_lin_1 = 0.5
    dropout_lin_rest = 0.5

    positive_class_weight = 4.0

    input_list_size = 2000

    param_dict = {}

    device = torch.device("cuda:" + str(gpu_nr) if torch.cuda.is_available() else "cpu")

    rooot = "/data/test_ril_seq_dir_a2_07_10_22/"

    path_to_here = os.getcwd()
    path_to_raw = path_to_here + rooot + "raw/"
    path_to_processed = path_to_here + rooot + "processed/"

    content_raw = os.listdir(path_to_raw)
    dataset = data_prep.MyDataset(root="." + rooot, input_list_size=input_list_size, content_raw=content_raw,
                                  path_to_processed=path_to_processed)

    # torch.manual_seed(12345) # necessary?
    torch.manual_seed(1234)
    dataset = dataset.shuffle()

    # for the RIL-seq data, because some graphs are to large:
    idx_smaller_graphs = []
    for i, graph in enumerate(dataset):
        if len(graph.x) <= max_graph_size:
            idx_smaller_graphs.append(i)

    dataset = dataset[idx_smaller_graphs]

    train_dataset, validation_dataset, test_dataset = stratified_train_val_test_split(dataset, train_proportion=0.7,
                                                                                      val_proportion=0.15,
                                                                                      test_proportion=0.15)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(validation_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    if upsampling:
        train_loader = utils.custom_loader(train_dataset, batch_size=batch_size,
                                           pos_class_weight=positive_class_weight)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    run = 0
    auprc_to_beat = minimum_auprc
    good_models = 0
    while run < number_of_runs:
        if more_modified:
            model = models.moremodifiedPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features,
                                              activation_funct)
        else:
            model = models.PNAnet4L(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
            #model = models.sharedParamPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
            #model = models.modifiedPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        pos_intances = 0
        neg_instances = 0
        for data in train_dataset:
            if int(data.y) == 0:
                neg_instances = neg_instances + 1
            elif int(data.y) == 1:
                pos_intances = pos_intances + 1
        print(f"\n"
              f"Number positive instances: {pos_intances}\n"
              f"Number negative instances: {neg_instances}\n")

        run = run + 1
        weights = torch.tensor([1, positive_class_weight]).to(device)
        if binary_classification:
            if upsampling:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_class_weight).to(device))
        else:
            if upsampling:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=weights)

        lmeans = []
        train_accs = []
        train_precs = []
        train_recs = []
        train_specs = []
        train_f1s = []
        train_aurpcs = []
        val_accs = []
        val_precs = []
        val_recs = []
        val_specs = []
        val_f1s = []
        val_aurpcs = []

        for epoch in range(epochs):
            print(f"\n"
                  f"Epoch {epoch}: \n")
            lmean = train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer,
                          more_modified, binary_classification)
            lmeans.append(lmean)
            # For evaluation, dropout is set to zero!:
            train_acc, train_prec, train_rec, train_spec, train_f1, train_aurpc = test(train_loader, 0, 0, model,
                                                                                       device, more_modified,
                                                                                       binary_classification)
            val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = test(validation_loader, 0, 0, model, device,
                                                                           more_modified, binary_classification)
            train_accs.append(train_acc)
            train_precs.append(train_prec)
            train_recs.append(train_rec)
            train_specs.append(train_spec)
            train_f1s.append(train_f1)
            train_aurpcs.append(train_aurpc)
            val_accs.append(val_acc)
            val_precs.append(val_prec)
            val_recs.append(val_rec)
            val_specs.append(val_spec)
            val_f1s.append(val_f1)
            val_aurpcs.append(val_aurpc)

            print(f"mean loss: {lmean} \n"
                  f"Training: \n"
                  f"acc: {train_acc}, prec: {train_prec}, rec: {train_rec}, spec: {train_spec}, f1: {train_f1}, aurpc: {train_aurpc} \n"
                  f"Validation: \n"
                  f"acc: {val_acc}, prec: {val_prec}, rec: {val_rec}, spec: {val_spec}, f1: {val_f1}, aurpc: {val_aurpc}")

            if val_aurpc > auprc_to_beat:
                path_model_state = path_to_here + "/model_state_" + file_name_base + "_run_" + str(run) + "_epoch_" + \
                                   str(epoch) + "_val_auprc_" + str("%.4f" % val_aurpc) + "_" + date + ".tar"
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "deg": deg
                            }, path_model_state)
                auprc_to_beat = val_aurpc
                good_models = good_models + 1

        # Save the metrics once for each full training of a model:
        vis_file_name = "output_for_vis_" + file_name_base + "_run_" + str(run) + "_" + date + ".pkl"
        track_metrics = {"lmeans": lmeans,
                         "param_dict": param_dict,
                         "train_accs": train_accs,
                         "train_precs": train_precs,
                         "train_recs": train_recs,
                         "train_specs": train_specs,
                         "train_f1s": train_f1s,
                         "train_aurpcs": train_aurpcs,
                         "val_accs": val_accs,
                         "val_precs": val_precs,
                         "val_recs": val_recs,
                         "val_specs": val_specs,
                         "val_f1s": val_f1s,
                         "val_aurpcs": val_aurpcs
                         }
        with open(vis_file_name, "wb") as pickle_out:
            rick.dump(track_metrics, pickle_out)

    print("done")



def main():
    save_model_state = True
    file_name_model_state = "/model_state_run_a_1_25_09_22.tar"

    output_file = "output_for_vis_run_a_1_25_09_22.pkl"

    more_modified = False

    epochs = 201
    batch_size = 10

    max_graph_size = 700

    given_train_val_split = False

    # Activation function:
    activation_funct = ReLU()

    # Optimizer:

    lr = 0.00005598
    #lr = 0.0001

    weight_decay = 0.00001471
    #weight_decay = 0.0001
    #dropout_conv_1_2 = 0.033686
    #dropout_conv_rest = 0.001152
    dropout_conv_1_2 = 0.033686
    dropout_conv_rest = 0.001152
    #dropout_lin_1 = 0.297727
    #dropout_lin_rest = 0.011801

    dropout_lin_1 = 0.5
    dropout_lin_rest = 0.5

    pos_weight = 4.0

    input_list_size = 2000

    gpu_nr = "cuda:0"
    device = torch.device(gpu_nr if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    conv_layer_type = "PNAConv"
    # pooling = "global_add (concatenation after each layer)"
    pooling = "global_mean + global_max (last layer)"

    #rooot = "/data_github/test_ril_seq/"
    #rooot = "/data/test_ril_seq/"
    #rooot = "/data/test_ril_seq_fixed_acc/"
    #rooot = "/data/test_ril_seq_17_08_22/"
    #rooot = "/data/test_ril_seq_all_17_08_22/"
    #rooot = "/data/test_ril_seq_18_08_22/"
    #rooot = "/data/test_ril_seq_all_18_08_22/"
    #rooot = "/data/test_test_23_08_22/"
    rooot = "/data/test_ril_seq_dir_a_20_09_22/"

    param_dict = {}

    # Start:

    path_to_here = os.getcwd()
    path_to_raw = path_to_here + rooot + "raw/"
    path_to_processed = path_to_here + rooot + "processed/"

    content_raw = os.listdir(path_to_raw)
    dataset = data_prep.MyDataset(root="." + rooot, input_list_size=input_list_size, content_raw=content_raw,
                                  path_to_processed=path_to_processed)

    # torch.manual_seed(12345) # necessary?
    torch.manual_seed(1234)
    dataset = dataset.shuffle()

    # for the RIL-seq data, because some graphs are to large:
    idx_smaller_graphs = []
    for i, graph in enumerate(dataset):
        if len(graph.x) <= max_graph_size:
            idx_smaller_graphs.append(i)

    dataset = dataset[idx_smaller_graphs]

    # Splitting in train and validation data:

    #if given_train_val_split == True:  # Use, when validation data is prepared:
    #    train_idxs = []
    #    val_idxs = []
    #    for ind, data in enumerate(dataset):
    #        split = int(data.split)
    #        if split == 0:
    #            train_idxs.append(ind)
    #        elif split == 1:
    #            val_idxs.append(ind)
    #    train_dataset = dataset[train_idxs]
    #    validation_dataset = dataset[val_idxs]
    # elif given_train_val_split == False:
    #    where_to_slice = round(len(dataset) * 0.8)
    #    test_dataset = dataset[where_to_slice:data_size]   # Splitting off of test dataset
    #    where_to_slice_rest = round(where_to_slice * 0.8)   # splitting the rest in training and validation datasets
    #    train_dataset = dataset[:where_to_slice]
    #    validation_dataset = dataset[where_to_slice:]
    #elif given_train_val_split == False:
    #    where_to_slice = round(len(dataset) * 0.8)
    #    test_dataset = dataset[where_to_slice:]  # Splitting off of test dataset
    #    where_to_slice_rest = round(where_to_slice * 0.8)  # splitting the rest in training and validation datasets
    #    train_dataset = dataset[:where_to_slice_rest]
    #    validation_dataset = dataset[where_to_slice_rest:where_to_slice]

    train_dataset, validation_dataset, test_dataset = stratified_train_val_test_split(dataset, train_proportion=0.7,
                                                                                      val_proportion=0.15,
                                                                                      test_proportion=0.15)


    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(validation_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                         shuffle=False, drop_last=True)

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    if more_modified:
        model = models.moremodifiedPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
    else:
        model = models.PNAnet4L(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_intances = 0
    neg_instances = 0
    for data in train_dataset:
        if int(data.y) == 0:
            neg_instances = neg_instances + 1
        elif int(data.y) == 1:
            pos_intances = pos_intances + 1
    print(f"\n"
          f"Number positive instances: {pos_intances}\n"
          f"Number negative instances: {neg_instances}\n")

    # for CrossEntropyLoss:
    #pos_weight = neg_instances / pos_intances
    weights = torch.tensor([1, pos_weight]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    lmeans = []
    train_accs = []
    train_precs = []
    train_recs = []
    train_specs = []
    train_f1s = []
    train_aurpcs = []
    val_accs = []
    val_precs = []
    val_recs = []
    val_specs = []
    val_f1s = []
    val_aurpcs = []

    #train_metrics = []
    #val_metrics = []

    for epoch in range(epochs):
        print(f"\n"
              f"Epoch {epoch}: \n")
        lmean = train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer,
                      more_modified)
        lmeans.append(lmean)
        # For evaluation, dropout is set to zero!:
        train_acc, train_prec, train_rec, train_spec, train_f1, train_aurpc = test(train_loader, 0, 0, model, device,
                                                                                   more_modified)
        val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = test(validation_loader, 0, 0, model, device,
                                                                       more_modified)
        train_accs.append(train_acc)
        train_precs.append(train_prec)
        train_recs.append(train_rec)
        train_specs.append(train_spec)
        train_f1s.append(train_f1)
        train_aurpcs.append(train_aurpc)
        val_accs.append(val_acc)
        val_precs.append(val_prec)
        val_recs.append(val_rec)
        val_specs.append(val_spec)
        val_f1s.append(val_f1)
        val_aurpcs.append(val_aurpc)

        #train_metrics.append([train_acc, train_prec, train_rec, train_spec, train_f1])
        #val_metrics.append([val_acc, val_prec, val_rec, val_spec, val_f1])
        print(f"mean loss: {lmean} \n"
              f"Training: \n"
              f"acc: {train_acc}, prec: {train_prec}, rec: {train_rec}, spec: {train_spec}, f1: {train_f1}, aurpc: {train_aurpc} \n"
              f"Validation: \n"
              f"acc: {val_acc}, prec: {val_prec}, rec: {val_rec}, spec: {val_spec}, f1: {val_f1}, aurpc: {val_aurpc}")
    track_metrics = {"lmeans": lmeans,
                     "param_dict": param_dict,
                     "train_accs": train_accs,
                     "train_precs": train_precs,
                     "train_recs": train_recs,
                     "train_specs": train_specs,
                     "train_f1s": train_f1s,
                     "train_aurpcs": train_aurpcs,
                     "val_accs": val_accs,
                     "val_precs": val_precs,
                     "val_recs": val_recs,
                     "val_specs": val_specs,
                     "val_f1s": val_f1s,
                     "val_aurpcs": val_aurpcs
                     }
    with open(output_file, "wb") as pickle_out:
        rick.dump(track_metrics, pickle_out)
    print(output_file)
    if save_model_state:
        path_model_state = path_to_here + file_name_model_state
        # torch.save(model.state_dict(), path_model_state)
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "deg": deg,
                    "dropout_lin_1": dropout_lin_1,
                    "dropout_lin_rest": dropout_lin_rest,
                    "activation_funct": activation_funct
                    }, path_model_state)

    #return track_metrics


if __name__ == "__main__":
    main()


def load_and_split(safe=True):
    rooot = "/data/test_ril_seq_dir_xy_21_11_22/"
    input_list_size = 2000
    file_interactions_overview_df = "data/test_ril_seq_dir_xy_21_11_22/interactions_overview_df.pkl"
    if safe:
        dataset_file = "data/test_ril_seq_dir_xy_21_11_22/train_val_test_datasets.pkl"

    path_to_here = os.getcwd()
    path_to_raw = path_to_here + rooot + "raw/"
    path_to_processed = path_to_here + rooot + "processed/"

    content_raw = os.listdir(path_to_raw)
    dataset = data_prep_b.MyDataset(root="." + rooot, input_list_size=input_list_size, content_raw=content_raw,
                                    interaction_index=True, path_to_processed=path_to_processed)

    torch.manual_seed(1234)

    pickle_in = open(file_interactions_overview_df, "rb")
    overview_df = rick.load(pickle_in)
    pickle_in.close()

    # match the graphs to the entries of the overview_df:
    if not overview_df.columns.str.contains("graph_index").any():
        overview_df["graph_index"] = "NA"
        for index, graph in tqdm(enumerate(dataset), total=len(dataset)):
            interaction_idx = int(graph.interaction_index)
            mask = overview_df["interaction_index"] == interaction_idx
            overview_df.loc[mask, "graph_index"] = index


    # remove interactions with "no favorable interaction". no graphs exist for these interactions:
    mask = overview_df["no_favorable_interaction"] == False
    clean_df = overview_df[mask].reset_index(drop=True)

    # check, whether there are entries with no graph
    mask = clean_df["graph_index"] == "NA"
    clean_df = clean_df[~mask].reset_index(drop=True)

    # split in training, validation and testing data:
    unique_rna_2s = clean_df.value_counts("rna_2")
    sampled_val_rna_2s = unique_rna_2s.sample(15, random_state=321)
    val_rna_2_list = list(sampled_val_rna_2s.index)
    val_rna_2_list = val_rna_2_list + ["spf", "omrA"]
    val_idxs = []
    for rna in tqdm(val_rna_2_list, total=len(val_rna_2_list)):
        idxs = list(clean_df[clean_df["rna_2"] == rna].index)
        val_idxs = val_idxs + idxs

    val_df = clean_df.loc[val_idxs].reset_index(drop=True)
    train_df = clean_df.drop(val_idxs).reset_index(drop=True)

    test_rna_2_list = ["ryhB", "sgrS"]
    test_idxs = []
    for rna in tqdm(test_rna_2_list, total=len(test_rna_2_list)):
        idxs = list(train_df[train_df["rna_2"] == rna].index)
        test_idxs = test_idxs + idxs

    test_df = train_df.loc[test_idxs].reset_index(drop=True)
    train_df = train_df.drop(test_idxs).reset_index(drop=True)

    trains = list(train_df["graph_index"])
    train_dataset = dataset[trains]
    vals = list(val_df["graph_index"])
    validation_dataset = dataset[vals]
    tests = list(test_df["graph_index"])
    test_dataset = dataset[tests]
    datasets = {"train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
                "train_dataset": train_dataset,
                "validation_dataset": validation_dataset,
                "test_dataset": test_dataset}
    if safe:
        with open(dataset_file, "wb") as pickle_out:
            rick.dump(datasets, pickle_out)

    return train_df, val_df, test_df, train_dataset, validation_dataset, test_dataset


def train_and_safe(load=True, load_degree=True, minimum_auprc=0.3, number_of_runs=1, gpu_nr=0, more_modified=False,
                   binary_classification=False, upsampling=True):
    if load:
        datasets_file = "data/test_ril_seq_dir_xy_21_11_22/train_val_test_datasets.pkl"
        with open(datasets_file, "rb") as pickle_in:
            datasets = rick.load(pickle_in)
        train_df = datasets["train_df"]
        val_df = datasets["val_df"]
        test_df = datasets["test_df"]
        train_dataset = datasets["train_dataset"]
        validation_dataset = datasets["validation_dataset"]
        test_dataset = datasets["test_dataset"]
    else:
        train_df, val_df, test_df, train_dataset, validation_dataset, test_dataset = load_and_split()

    file_name_base = "ril_seq_run_xy_8"
    date = "09_12_22"

    deg_hist_file = "data/test_ril_seq_dir_xy_21_11_22/train_datasets_deg_hist.pkl"

    path_to_here = os.getcwd()

    epochs = 50
    batch_size = 10

    #activation_funct = ReLU()
    activation_funct = LeakyReLU()
    lr = 0.0001
    weight_decay = 0.0001
    dropout_conv_1_2 = 0.4
    ##
    dropout_conv_rest = 0.001152  # not used, just a relict...
    ##
    dropout_lin_1 = 0.5
    dropout_lin_rest = 0.5

    input_list_size = 2000

    param_dict = {}

    device = torch.device("cuda:" + str(gpu_nr) if torch.cuda.is_available() else "cpu")

    train_balance = train_df.value_counts("label")[0] / train_df.value_counts("label")[1]
    val_balance = val_df.value_counts("label")[0] / val_df.value_counts("label")[1]
    test_balance = test_df.value_counts("label")[0] / test_df.value_counts("label")[1]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(validation_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f"balance = negative instances / positive instances:\n"
          f"train balance: {train_balance}, \n"
          f"val balance: {val_balance}, \n"
          f"test balance: {test_balance}")

    if upsampling:
        train_loader = utils.custom_loader(train_dataset, batch_size=batch_size,
                                           pos_class_weight=train_balance)
        eval_train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        eval_train_loader = train_loader

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

    if not load_degree:
        # Compute the maximum in-degree in the training data.
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        with open(deg_hist_file, "wb") as pickle_out:
            rick.dump(deg, pickle_out)
    else:
        with open(deg_hist_file, "rb") as pickle_in:
            deg = rick.load(pickle_in)

    run = 0
    auprc_to_beat = minimum_auprc
    good_models = 0
    while run < number_of_runs:
        if more_modified:
            model = models.moremodifiedPNAnet(dropout_lin_1, dropout_lin_rest, deg, train_dataset.num_node_features,
                                              activation_funct)
        else:
            # model = models.PNAnet4L(dropout_lin_1, dropout_lin_rest, deg, train_dataset.num_node_features, activation_funct)
            model = models.PNAnet6L(dropout_lin_1, dropout_lin_rest, deg, train_dataset.num_node_features,
                                    activation_funct)
            # model = models.sharedParamPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
            # model = models.modifiedPNAnet(dropout_lin_1, dropout_lin_rest, deg, dataset.num_node_features, activation_funct)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        run = run + 1
        #weights = torch.tensor([1, positive_class_weight]).to(device)
        if binary_classification:
            if upsampling:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_balance).to(device))
        else:
            if upsampling:
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 0.9]).to(device))
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, float(train_balance)]).to(device))

        lmeans = []
        train_accs = []
        train_precs = []
        train_recs = []
        train_specs = []
        train_f1s = []
        train_aurpcs = []
        val_accs = []
        val_precs = []
        val_recs = []
        val_specs = []
        val_f1s = []
        val_aurpcs = []
        test_accs = []
        test_precs = []
        test_recs = []
        test_specs = []
        test_f1s = []
        test_aurpcs = []

        # Look at baseline before training!:
        #train_acc, train_prec, train_rec, train_spec, train_f1, train_aurpc = test(eval_train_loader, 0, 0, model,
        #                                                                           device, more_modified,
        #                                                                           binary_classification)
        val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = test(validation_loader, 0, 0, model, device,
                                                                       more_modified, binary_classification)
        test_acc, test_prec, test_rec, test_spec, test_f1, test_aurpc = test(test_loader, 0, 0, model, device,
                                                                             more_modified, binary_classification)
        #train_accs.append(train_acc)
        #train_precs.append(train_prec)
        #train_recs.append(train_rec)
        #train_specs.append(train_spec)
        #train_f1s.append(train_f1)
        #train_aurpcs.append(train_aurpc)
        val_accs.append(val_acc)
        val_precs.append(val_prec)
        val_recs.append(val_rec)
        val_specs.append(val_spec)
        val_f1s.append(val_f1)
        val_aurpcs.append(val_aurpc)
        test_accs.append(test_acc)
        test_precs.append(test_prec)
        test_recs.append(test_rec)
        test_specs.append(test_spec)
        test_f1s.append(test_f1)
        test_aurpcs.append(test_aurpc)

        print(f"Baseline before training:\n")
        print("Validation: \n"
              f"acc: {val_acc}, prec: {val_prec}, rec: {val_rec}, spec: {val_spec}, f1: {val_f1}, aurpc: {val_aurpc} \n"
              f"test: \n"
              f"acc: {test_acc}, prec: {test_prec}, rec: {test_rec}, spec: {test_spec}, f1: {test_f1}, aurpc: {test_aurpc}")

        for epoch in range(epochs):
            print(f"\n"
                  f"Epoch {epoch}: \n")
            lmean = train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer,
                          more_modified, binary_classification)
            lmeans.append(lmean)
            # For evaluation, dropout is set to zero!:
            if epoch % 10 == 0:
                train_acc, train_prec, train_rec, train_spec, train_f1, train_aurpc = test(eval_train_loader, 0, 0, model,
                                                                                           device, more_modified,
                                                                                           binary_classification)
                train_accs.append(train_acc)
                train_precs.append(train_prec)
                train_recs.append(train_rec)
                train_specs.append(train_spec)
                train_f1s.append(train_f1)
                train_aurpcs.append(train_aurpc)
            val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = test(validation_loader, 0, 0, model, device,
                                                                           more_modified, binary_classification)
            test_acc, test_prec, test_rec, test_spec, test_f1, test_aurpc = test(test_loader, 0, 0, model, device,
                                                                           more_modified, binary_classification)
            val_accs.append(val_acc)
            val_precs.append(val_prec)
            val_recs.append(val_rec)
            val_specs.append(val_spec)
            val_f1s.append(val_f1)
            val_aurpcs.append(val_aurpc)
            test_accs.append(test_acc)
            test_precs.append(test_prec)
            test_recs.append(test_rec)
            test_specs.append(test_spec)
            test_f1s.append(test_f1)
            test_aurpcs.append(test_aurpc)

            if epoch % 10 == 0:
                print(f"Training: \n"
                      f"acc: {train_acc}, prec: {train_prec}, rec: {train_rec}, spec: {train_spec}, f1: {train_f1}, aurpc: {train_aurpc} \n")
            print(f"mean loss: {lmean} \n"
                  f"Validation: \n"
                  f"acc: {val_acc}, prec: {val_prec}, rec: {val_rec}, spec: {val_spec}, f1: {val_f1}, aurpc: {val_aurpc} \n"
                  f"test: \n"
                  f"acc: {test_acc}, prec: {test_prec}, rec: {test_rec}, spec: {test_spec}, f1: {test_f1}, aurpc: {test_aurpc}")

            if val_aurpc > auprc_to_beat:
                path_model_state = path_to_here + "/model_state_" + file_name_base + "_run_" + str(run) + "_epoch_" + \
                                   str(epoch) + "_val_auprc_" + str("%.4f" % val_aurpc) + "_" + date + ".tar"
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "deg": deg
                            }, path_model_state)
                auprc_to_beat = val_aurpc
                good_models = good_models + 1

        # Save the metrics once for each full training of a model:
        vis_file_name = "output_for_vis_" + file_name_base + "_run_" + str(run) + "_" + date + ".pkl"
        track_metrics = {"lmeans": lmeans,
                         "param_dict": param_dict,
                         "train_accs": train_accs,
                         "train_precs": train_precs,
                         "train_recs": train_recs,
                         "train_specs": train_specs,
                         "train_f1s": train_f1s,
                         "train_aurpcs": train_aurpcs,
                         "val_accs": val_accs,
                         "val_precs": val_precs,
                         "val_recs": val_recs,
                         "val_specs": val_specs,
                         "val_f1s": val_f1s,
                         "val_aurpcs": val_aurpcs,
                         "test_accs": test_accs,
                         "test_precs": test_precs,
                         "test_recs": test_recs,
                         "test_specs": test_specs,
                         "test_f1s": test_f1s,
                         "test_aurpcs": test_aurpcs
                         }
        with open(vis_file_name, "wb") as pickle_out:
            rick.dump(track_metrics, pickle_out)

    print("done")



