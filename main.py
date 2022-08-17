import os
import os.path as osp
import pickle as rick
import statistics
import torch
from torch.nn import ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from sklearn.metrics import precision_recall_curve, auc


# Helper scripts:
import data_prep
import models


def train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer):
    model.train()
    batch_nr = 0
    losses = 0
    for batch in train_loader:
        batch_nr = batch_nr + 1
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch,
                    dropout_conv_1_2, dropout_conv_rest)
        loss = criterion(out, batch.y)   # for CrossEntropyLoss
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


def test(loader, dropout_conv_1_2, dropout_conv_rest, model, device):
    model.eval()
    test_predictions = []
    proba_positiv_prediction = []
    test_labels = []
    batches = 0
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch,
                        dropout_conv_1_2, dropout_conv_rest)
            pred = out.detach()
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
            pred_probas = out.softmax(dim=1).detach().cpu()
            proba_ones = pred_probas[:, 1].tolist()  # Only the probability of class one is needed for the AUPRC (softmax
                                            # probabilities add up to 100 % so no information is lost...)
            proba_positiv_prediction = proba_positiv_prediction + proba_ones
        # mae = calculate_mae(test_predictions, test_labels)
    acc, prec, rec, spec, f1 = metrics(test_predictions, test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels, proba_positiv_prediction)
    area_under_rpc = auc(recall, precision)
    return acc, prec, rec, spec, f1, area_under_rpc


def load_trained_model(file_name_model_state, device, num_node_features, model_name=models.PNAnet4L):
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


def main():
    save_model_state = True
    file_name_model_state = "/model_state_play_arounds_run_9_17_08_22.tar"

    output_file = "output_for_vis_play_arounds_run_9_17_08_22.pkl"
    epochs = 201
    batch_size = 10

    max_graph_size = 550

    given_train_val_split = False

    # Activation function:
    activation_funct = ReLU()

    # Optimizer:

    lr = 0.00005598

    #weight_decay = 0.00001471
    weight_decay = 0.0001
    #dropout_conv_1_2 = 0.033686
    #dropout_conv_rest = 0.001152
    dropout_conv_1_2 = 0.3
    dropout_conv_rest = 0.3
    #dropout_lin_1 = 0.297727
    #dropout_lin_rest = 0.011801

    dropout_lin_1 = 0.5
    dropout_lin_rest = 0.5

    input_list_size = 2000

    gpu_nr = "cuda:2"
    device = torch.device(gpu_nr if torch.cuda.is_available() else "cpu")

    conv_layer_type = "PNAConv"
    # pooling = "global_add (concatenation after each layer)"
    pooling = "global_mean + global_max (last layer)"

    #rooot = "/data_github/test_ril_seq/"
    #rooot = "/data/test_ril_seq/"
    #rooot = "/data/test_ril_seq_fixed_acc/"
    rooot = "/data/test_ril_seq_16_08_22/"

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

    # for CrossEntropyLoss:
    pos_weight = neg_instances / pos_intances
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
        lmean = train(train_loader, criterion, dropout_conv_1_2, dropout_conv_rest, model, device, optimizer)
        lmeans.append(lmean)
        # For evaluation, dropout is set to zero!:
        train_acc, train_prec, train_rec, train_spec, train_f1, train_aurpc = test(train_loader, 0, 0, model, device)
        val_acc, val_prec, val_rec, val_spec, val_f1, val_aurpc = test(validation_loader, 0, 0, model, device)
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






