import torch
from torch_geometric.nn import BatchNorm, PNAConv
from torch_geometric.utils import degree
from torch.nn import Dropout
from torch_geometric.utils import dropout_adj
from torch.nn import ModuleList
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ReLU
from torch.nn import LeakyReLU


class PNAnet4L(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct):
        super().__init__()

        self.activation_funct = activation_funct

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

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

        self.lin1 = Linear((128 * 2), 128)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin3 = Linear(64, 2)    # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, dropout_conv_1_2, dropout_conv_rest):
        # Convolutions:

        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv1(x, dropped_edge_index_1, dropped_edge_attr_1)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        dropped_edge_index_2, dropped_edge_attr_2 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv2(x, dropped_edge_index_2, dropped_edge_attr_2)
        #if self.training == True:
        #    x = x / conv_dropout  # To implement inverted dropout (so no scaling is necessary during testing)
        # -> not necessary, inverted dropout should be implemented automatically.
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        dropped_edge_index_3, dropped_edge_attr_3 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv3(x, dropped_edge_index_3, dropped_edge_attr_3)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        dropped_edge_index_4, dropped_edge_attr_4 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv4(x, dropped_edge_index_4, dropped_edge_attr_4)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)
        x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        x = self.dropout_lin2(x)
        x = self.lin3(x)
        return x


class PNAnet6L(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, activation_funct):
        super().__init__()

        self.activation_funct = activation_funct

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.conv1 = PNAConv(in_channels=dataset.num_node_features, out_channels=128,
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

        self.lin1 = Linear((128 * 8), 896)
        #self.lin1 = Linear(((128 * 2) + 1), 128)
        if self.training == True:
            self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(896, 384)
        if self.training == True:
            self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        # Introduce IntaRNA energy in a later layer, to give it more importance.
        self.lin3 = Linear((384 + 1), 64)
        if self.training == True:
            self.dropout_lin3 = Dropout(p=dropout_lin_rest)
        self.lin4 = Linear(64, 2)    # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, dropout_conv_1_2, dropout_conv_rest):
        # Convolutions:

        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv1(x, dropped_edge_index_1, dropped_edge_attr_1)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        x1 = global_add_pool(x, batch)

        dropped_edge_index_2, dropped_edge_attr_2 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv2(x, dropped_edge_index_2, dropped_edge_attr_2)
        #if self.training == True:
        #    x = x / conv_dropout  # To implement inverted dropout (so no scaling is necessary during testing)
        # -> not necessary, inverted dropout should be implemented automatically.
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        x2 = global_add_pool(x, batch)

        dropped_edge_index_3, dropped_edge_attr_3 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv3(x, dropped_edge_index_3, dropped_edge_attr_3)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        x3 = global_add_pool(x, batch)

        dropped_edge_index_4, dropped_edge_attr_4 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv4(x, dropped_edge_index_4, dropped_edge_attr_4)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4 = global_add_pool(x, batch)

        dropped_edge_index_5, dropped_edge_attr_5 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv5(x, dropped_edge_index_5, dropped_edge_attr_5)
        x = self.batch_norm5(x)
        x = self.activation_funct(x)

        x5 = global_add_pool(x, batch)

        dropped_edge_index_6, dropped_edge_attr_6 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv6(x, dropped_edge_index_6, dropped_edge_attr_6)
        x = self.batch_norm6(x)
        x = self.activation_funct(x)

        x6a = global_add_pool(x, batch)
        x6b = global_max_pool(x, batch)
        x6c = global_mean_pool(x, batch)

        # Linears:

        en = intarna_energy.reshape(-1, 1)

        x = torch.cat([x1, x2, x3, x4, x5, x6a, x6b, x6c], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)
        x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        x = self.dropout_lin2(x)

        x = torch.cat([x, en], dim=1)

        x = self.lin3(x)
        x = self.activation_funct(x)
        x = self.dropout_lin3(x)
        x = self.lin4(x)
        return x