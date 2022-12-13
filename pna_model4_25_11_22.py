import os
import os.path as osp
import pickle as rick
import statistics

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataLoader

from torch_geometric.utils import degree
from torch_geometric.nn import BatchNorm, PNAConv, GCNConv
from torch.nn import ModuleList
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ReLU
from torch.nn import LeakyReLU

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from models import modifiedPNAConv
#import utils


class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # root = Where the Dataset should be stored. This Folder is
        # split into raw_dir (where the raw data should be or is downloaded to)
        # and processed_dir (where the processed data is saved).
        # I use a path to the raw data iin the process function and don't
        # use the raw_dir.
        super().__init__(root, transform, pre_transform)
        # It can be that it should be:
        # super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return content_raw

    @property
    def processed_file_names(self):
        content_processed = os.listdir(path_to_processed)
        if len(content_processed) > 10:
            file_names = content_processed
            file_names.remove("pre_filter.pt")
            file_names.remove("pre_transform.pt")
        else:
            file_names = ["data_1.pt", "data_2.pt"]
        return file_names

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print(raw_path)   # debugging
            pickle_in = open(raw_path, "rb")
            for step in range(0, input_list_size):
                try:
                    entry = rick.load(pickle_in)
                    #print(entry[0][0])
                    #print("a")    # debugging
                    #feat = entry[0][0]
                    #for node_feat in feat:
                    #    node_feat[4] = node_feat[4] * 0.01
                    #features = torch.tensor(feat, dtype=torch.float)
                    features = torch.tensor(entry[0][0], dtype=torch.float)
                    #print("aa")    # debugging
                    edge_index = torch.tensor(entry[0][1], dtype=torch.long)
                    #print("aaa")    # debugging
                    edge_index = edge_index.t().contiguous()
                    #print("aaaa")    # debugging
                    edgetype = []
                    #print("b")    # debugging
                    for index_tuple in entry[0][1]:
                        start_node = index_tuple[0]
                        end_node = index_tuple[1]
                        if start_node <= 78 and end_node <= 78:
                            if start_node == (end_node - 1) or start_node == (end_node + 1):
                                edgetype.append([1, 0, 0])  # covalent bond
                            else:
                                edgetype.append([0, 1, 0])  # intra-molecular bond
                        elif start_node >= 79 and end_node >= 79:
                            if start_node == (end_node - 1) or start_node == (end_node + 1):
                                edgetype.append([1, 0, 0])  # covalent bond
                            else:
                                edgetype.append([0, 1, 0])  # intra-molecular bond
                        else:
                            edgetype.append([0, 0, 1])  # inter-molecular bond
                    edge_type_tensor = torch.tensor(edgetype)
                    #print("c")    # debugging
                    # edge_type_tensor = edge_type_tensor.reshape([-1, 1])
                    weights = torch.tensor(entry[0][2])
                    weights = weights.reshape([-1, 1])
                    weights = weights.float()  # necessary, because edge_attr must be in different
                    # shape, than edge_weight
                    edge_attr = torch.cat((edge_type_tensor, weights), dim=1)
                    label = entry[0][3]
                    label_tensor = torch.tensor([label], dtype=torch.float)
                    data_entry = Data(x=features,
                                      edge_index=edge_index,
                                      edge_attr=edge_attr,
                                      y=label_tensor
                                      )
                    #print("d")    # debugging
                    torch.save(data_entry,
                               osp.join(self.processed_dir,
                                        f'data_{idx}.pt'))
                    idx = idx + 1
                    #print("e")    # debugging
                except:
                    #print("break")    # debugging
                    break
            pickle_in.close()
        idx = idx - 1
        print(idx)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   f'data_{idx}.pt'))
        return data


class PNAnet(torch.nn.Module):
    def __init__(self, device, batch_size):
        super().__init__()

        self.add_rna_id = False
        self.set2set = False
        self.device = device
        self.batch_size = batch_size

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation", "linear"]
        if self.add_rna_id:
            num_node_features = 6
        else:
            num_node_features = 4

        self.conv1 = PNAConv(in_channels=num_node_features, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm1 = BatchNorm(128)

        self.conv2 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm2 = BatchNorm(128)

        self.conv3 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm3 = BatchNorm(128)

        self.conv4 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm4 = BatchNorm(128)

        self.conv5 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm5 = BatchNorm(128)

        self.conv6 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm6 = BatchNorm(128)

        if self.set2set:
            self.set2set_pooling = utils.set2set_aggr(in_channels=128, processing_steps=1)

        #self.lin1 = Linear((128 * 6), 128)
        self.lin1 = Linear((128 * 2), 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        if self.add_rna_id:
            id_dim_one = [1] * 135 + [0] * 79
            id_dim_one = id_dim_one * self.batch_size
            id_dim_one_tensor = torch.tensor(id_dim_one).unsqueeze(dim=1).to(self.device)
            id_dim_two = [0] * 135 + [1] * 79
            id_dim_two = id_dim_two * self.batch_size
            id_dim_two_tensor = torch.tensor(id_dim_two).unsqueeze(dim=1).to(self.device)
            #print(f"batch: {batch}")
            #print(f"x: {x.size()}")
            #print(f"id_dim_one_tensor: {id_dim_one_tensor.size()}")
            #print(f"id_dim_two_tensor: {id_dim_two_tensor.size()}")
            x = torch.concat([x, id_dim_one_tensor, id_dim_two_tensor], dim=1)

        # Convolutions:
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = activation_funct(x)
        #x1 = global_add_pool(x, batch)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = activation_funct(x)
        #x2 = global_add_pool(x, batch)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = activation_funct(x)
        #x3 = global_add_pool(x, batch)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = activation_funct(x)
        #x4 = global_add_pool(x, batch)
        x = self.conv5(x, edge_index, edge_attr)
        x = self.batch_norm5(x)
        x = activation_funct(x)
        #x5 = global_add_pool(x, batch)
        x = self.conv6(x, edge_index, edge_attr)
        x = self.batch_norm6(x)
        x = activation_funct(x)
        #x6 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        # Linears:
        #x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        if self.set2set:
            x = self.set2set_pooling(x, batch, dim_size=self.batch_size)
        else:
            x = torch.cat([x1, x2], dim=1)
        x = self.lin1(x)
        x = activation_funct(x)
        x = self.lin2(x)
        x = activation_funct(x)
        x = self.lin3(x)
        return x


class modifiedPNAnet(torch.nn.Module):
    def __init__(self, device, batch_size):
        super().__init__()

        self.add_rna_id = False
        self.device = device
        self.batch_size = batch_size

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation", "linear"]
        if self.add_rna_id:
            num_node_features = 6
        else:
            num_node_features = 4

        self.conv1 = modifiedPNAConv(in_channels=num_node_features, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm1 = BatchNorm(128)

        self.conv2 = modifiedPNAConv(in_channels=128, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm2 = BatchNorm(128)

        self.conv3 = modifiedPNAConv(in_channels=128, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm3 = BatchNorm(128)

        self.conv4 = modifiedPNAConv(in_channels=128, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm4 = BatchNorm(128)

        self.conv5 = modifiedPNAConv(in_channels=128, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm5 = BatchNorm(128)

        self.conv6 = modifiedPNAConv(in_channels=128, out_channels=128,
                                     aggregators=aggregators, scalers=scalers, deg=deg,
                                     edge_dim=3, towers=1, pre_layers=1, post_layers=1,
                                     divide_input=False)
        self.batch_norm6 = BatchNorm(128)

        #self.lin1 = Linear((128 * 6), 128)
        self.lin1 = Linear((128 * 2), 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        if self.add_rna_id:
            id_dim_one = [1] * 135 + [0] * 79
            id_dim_one = id_dim_one * self.batch_size
            id_dim_one_tensor = torch.tensor(id_dim_one).unsqueeze(dim=1).to(self.device)
            id_dim_two = [0] * 135 + [1] * 79
            id_dim_two = id_dim_two * self.batch_size
            id_dim_two_tensor = torch.tensor(id_dim_two).unsqueeze(dim=1).to(self.device)
            #print(f"batch: {batch}")
            #print(f"x: {x.size()}")
            #print(f"id_dim_one_tensor: {id_dim_one_tensor.size()}")
            #print(f"id_dim_two_tensor: {id_dim_two_tensor.size()}")
            x = torch.concat([x, id_dim_one_tensor, id_dim_two_tensor], dim=1)

        # Convolutions:
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = activation_funct(x)
        #x1 = global_add_pool(x, batch)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = activation_funct(x)
        #x2 = global_add_pool(x, batch)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = activation_funct(x)
        #x3 = global_add_pool(x, batch)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = activation_funct(x)
        #x4 = global_add_pool(x, batch)
        x = self.conv5(x, edge_index, edge_attr)
        x = self.batch_norm5(x)
        x = activation_funct(x)
        #x5 = global_add_pool(x, batch)
        x = self.conv6(x, edge_index, edge_attr)
        x = self.batch_norm6(x)
        x = activation_funct(x)
        #x6 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        # Linears:
        #x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = self.lin1(x)
        x = activation_funct(x)
        x = self.lin2(x)
        x = activation_funct(x)
        x = self.lin3(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # .num_classes are an attribute of
        # InMemoryDataset, but NOT of Dataset.

        self.conv1 = GCNConv(dataset.num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)
        #self.conv4 = GCNConv(128, 128)
        #self.conv5 = GCNConv(128, 128)
        #self.conv6 = GCNConv(128, 128)
        # if max, min, median -> change of input: x3 (x4?)
        self.lin1 = Linear((128 * 2), 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Only use the weights from the edge_attr:
        edge_weight = edge_attr[:, 3:]

        x = self.conv1(x, edge_index, edge_weight)
        x = activation_funct(x)
        x = self.conv2(x.float(), edge_index)
        x = activation_funct(x)
        x = self.conv3(x.float(), edge_index)
        #x = activation_funct(x)
        #x = self.conv4(x.float(), edge_index)
        #x = activation_funct(x)
        #x = self.conv5(x.float(), edge_index)
        #x = activation_funct(x)
        #x = self.conv6(x.float(), edge_index)
        # 2. Readout layer
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)
                     # max, min, median (additional concatenation of previous layers?)
        # 3. linear layers
        x = self.lin1(x)
        x = activation_funct(x)
        # x = F.dropout(x, p=0.1, training=self.training)   # try diff. values for p
        x = self.lin2(x)
        x = activation_funct(x)
        x = self.lin3(x)
        return x


def train(train_loader, criterion, loss_list):
    model.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        labels = batch.y.reshape([-1, 1])
        #weights = batch.edge_weight.reshape([-1, 1])
        #edge_weights = torch.tensor(weights, dtype=torch.float)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(float(loss))
    return loss_list


def calculate_mae(x, y):
    sum_of_errors = 0
    for i, j in enumerate(y):
        err = x[i] - j
        sum_of_errors = sum_of_errors + abs(err)
    mae = sum_of_errors / len(y)
    return mae


def test(loader):
    model.eval()
    test_predictions = []
    test_labels = []
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.detach()
            pred = pred.squeeze()
            t_labels = batch.y.detach()
            #print(f'Pred: {pred}, \n   Y: {t_labels}')
            test_predictions = test_predictions + pred.tolist()
            test_labels = test_labels + t_labels.tolist()
    mae = calculate_mae(test_predictions, test_labels)
    return test_predictions, test_labels, mae


# Hyperparameters:

val_mae_to_beat = 0.52

#data_size = 2000
#data_size = 43000
#data_size = 26110
data_size = 130000

epochs = 50

batch_size = 9

# Activation function:
#activation_funct = ReLU()
activation_funct = LeakyReLU()

# Optimizer:
#lr = 0.0003298
#lr = 0.0000123
#lr = 0.00123
lr = 0.000123

gpu_nr = "cuda:3"
device = torch.device(gpu_nr if torch.cuda.is_available() else "cpu")

conv_layer_type = "GCN"
#pooling = "global_add (concatenation after each layer)"
pooling = "global_mean + global_max (last layer)"

# Parameter dict:
param_dict = {"data_size": data_size,
              "Graph conv layer type": conv_layer_type,
              "epochs": epochs,
              "batch_size": batch_size,
              "learning_rate": lr,
              "activation_function": activation_funct,
              "device": gpu_nr,
              "Pooling": pooling
              }

file_name = "output_for_vis_poddar_gcn_sp2_2_clean_50ep_04_12_22.pkl"

#file_name_model_state = "/model_state_pnaconv4_optimized1_150ep_25_11_22_22.tar"

file_name_base = "poddar_gcn_sp2_2_clean_50ep_"
date = "04_12_22"

dataset_file = file_name_base + "_datasets_" + date + ".pkl"
safe = False
load_datasets_from_file = True
saved_dataset_file = "poddar_GCN_clean_50ep_datasets_29_11_22.pkl"

# start:

#rooot = "/test_pna_onehot_clean/"
rooot = "/test_pna_all/"
#rooot = "/test_pna_energy/"
#rooot = "/test_diff_energy/"
#rooot = "/test_pna_clean_energy/"
#rooot = "/test_naskulwar/"


input_list_size = 2000

path_to_here = os.getcwd()
path_to_raw = path_to_here + rooot + "raw/"
path_to_processed = path_to_here + rooot + "processed/"

content_raw = os.listdir(path_to_raw)
dataset = MyDataset(root="." + rooot)

if not load_datasets_from_file:
    # Remove interactions with integer-labels
    integer_idxs = []
    for i in tqdm(range(len(dataset))):
        if float(dataset[i].y) == 1:
            integer_idxs = integer_idxs + [i]
        elif float(dataset[i].y) == 2:
            integer_idxs = integer_idxs + [i]
        elif float(dataset[i].y) == 3:
            integer_idxs = integer_idxs + [i]
        elif float(dataset[i].y) == 4:
            integer_idxs = integer_idxs + [i]
        elif float(dataset[i].y) == 5:
            integer_idxs = integer_idxs + [i]

    non_integer_idx = [idx for idx in range(len(dataset)) if idx not in integer_idxs]

    dataset = dataset[non_integer_idx]
    # torch.manual_seed(12345) # necessary?
    torch.manual_seed(1234)
    dataset = dataset.shuffle()

    #where_to_slice = round(data_size * 0.8)
    where_to_slice = round(len(dataset) * 0.8)

    # Splitting off of test dataset
    test_dataset = dataset[where_to_slice:data_size]
    # splitting the rest in training and validation datasets
    where_to_slice_rest = round(where_to_slice * 0.8)
    train_dataset = dataset[:where_to_slice_rest]
    validation_dataset = dataset[where_to_slice_rest:where_to_slice]
else:
    with open(saved_dataset_file, "rb") as p_in:
        train_dataset = rick.load(p_in)
        validation_dataset = rick.load(p_in)
        test_dataset = rick.load(p_in)

if safe:
    with open(dataset_file, "wb") as p_out:
        rick.dump(train_dataset, p_out)
        rick.dump(validation_dataset, p_out)
        rick.dump(test_dataset, p_out)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(validation_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

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

model = PNAnet(device, batch_size)
#model = GCN()
#model = modifiedPNAnet(device, batch_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)      # weight_decay=5e-4
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.000001)
#criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()

lmeans = []
lmins = []
val_maes = []

acc_at_good_epochs_dic = {}

for epoch in range(epochs):
    loss_list = []
    losses = train(train_loader, criterion, loss_list)
    lmean = statistics.mean(losses)
    lmin = min(losses)
    lmeans.append(lmean)
    lmins.append(lmin)
    val_preds, val_labels, val_mae = test(validation_loader)
    val_maes.append(val_mae)
    #scheduler.step(val_mae)
    print(f'Epoch: {epoch:03d}, mean loss: {lmean}, val MAE: {val_mae}')

    if val_mae < val_mae_to_beat:
        path_model_state = path_to_here + "/model_state_" + file_name_base + "_epoch_" + \
                           str(epoch) + "_val_mae_" + str("%.4f" % val_mae) + "_" + date + ".tar"
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "deg": deg
                    }, path_model_state)
        val_mae_to_beat = val_mae

        # Test performance at this epoch:
        test_preds, test_labels, test_mae = test(test_loader)

        # For visualization (scatterplot of preds vs. labels):
        train_preds, train_labels, train_mae = test(train_loader)

        key1 = f"train_preds_epoch_{epoch}"
        key2 = f"train_labels_epoch_{epoch}"
        key3 = f"val_preds_epoch_{epoch}"
        key4 = f"val_labels_epoch_{epoch}"
        key5 = f"test_preds_epoch_{epoch}"
        key6 = f"test_labels_epoch_{epoch}"
        acc_at_good_epochs_dic[key1] = train_preds
        acc_at_good_epochs_dic[key2] = train_labels
        acc_at_good_epochs_dic[key3] = val_preds
        acc_at_good_epochs_dic[key4] = val_labels
        acc_at_good_epochs_dic[key5] = test_preds
        acc_at_good_epochs_dic[key6] = test_labels

acc_at_good_epochs_dic_path = path_to_here + "/acc_at_good_epochs_dic_" + file_name_base + date + ".pkl"
pickle_out = open(acc_at_good_epochs_dic_path, "wb")
rick.dump(acc_at_good_epochs_dic, pickle_out)
pickle_out.close()

path_model_state = path_to_here + "/model_state_" + file_name_base + "_epoch_" + \
                           str(epoch) + "_val_mae_" + str("%.4f" % val_mae) + "_" + date + ".tar"
# torch.save(model.state_dict(), path_model_state)
torch.save({"epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "deg": deg
            }, path_model_state)

train_predictions, train_labels, train_mae = test(train_loader)
val_predictions, val_labels, val_mae = test(validation_loader)

print("training done: ")
print(f'mean loss: {statistics.mean(lmeans)}, '
      f'min loss: {min(lmins)}, '
      f'train MAE: {train_mae}, '
      f'val MAE: {val_mae}')

# Save lists of mean loss, min loss and accuracy for later
# visualization
pickle_out = open(file_name, "wb")
rick.dump(lmeans, pickle_out)
rick.dump(lmins, pickle_out)
rick.dump(train_predictions, pickle_out)
rick.dump(train_labels, pickle_out)
rick.dump(val_predictions, pickle_out)
rick.dump(val_labels, pickle_out)
rick.dump(param_dict, pickle_out)
rick.dump(val_maes, pickle_out)
rick.dump(train_mae, pickle_out)
rick.dump(val_mae, pickle_out)

pickle_out.close()

print("done")
