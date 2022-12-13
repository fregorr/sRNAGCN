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
from datetime import datetime
import utils


class PNAnet4L(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct):
        super().__init__()

        self.activation_funct = activation_funct

        self.use_batch_norm = True
        self.binary_classification = False
        self.set2set = True

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        # Additional "linear" scaler could be useful. Especially higher-degree graphs seem to benefit.

        self.conv1 = PNAConv(in_channels=num_node_features, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        if self.use_batch_norm:
            self.batch_norm1 = BatchNorm(128, momentum=0.05)

        self.conv2 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        if self.use_batch_norm:
            self.batch_norm2 = BatchNorm(128, momentum=0.05)

        self.conv3 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        if self.use_batch_norm:
            self.batch_norm3 = BatchNorm(128, momentum=0.05)

        self.conv4 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        if self.use_batch_norm:
            self.batch_norm4 = BatchNorm(128, momentum=0.05)

        if self.set2set:
            self.set2set_pooling = utils.set2set_aggr(in_channels=128, processing_steps=1)

        self.lin1 = Linear((128 * 2), 128)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        if self.binary_classification:
            self.lin3 = Linear(64, 1)
        else:
            self.lin3 = Linear(64, 2)    # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2, dropout_conv_rest):
        # Convolutions:

        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv1(x, dropped_edge_index_1, dropped_edge_attr_1)
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        x = self.activation_funct(x)

        dropped_edge_index_2, dropped_edge_attr_2 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv2(x, dropped_edge_index_2, dropped_edge_attr_2)
        #if self.training == True:
        #    x = x / conv_dropout  # To implement inverted dropout (so no scaling is necessary during testing)
        # -> not necessary, inverted dropout should be implemented automatically.
        if self.use_batch_norm:
            x = self.batch_norm2(x)
        x = self.activation_funct(x)

        dropped_edge_index_3, dropped_edge_attr_3 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv3(x, dropped_edge_index_3, dropped_edge_attr_3)
        if self.use_batch_norm:
            x = self.batch_norm3(x)
        x = self.activation_funct(x)

        dropped_edge_index_4, dropped_edge_attr_4 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv4(x, dropped_edge_index_4, dropped_edge_attr_4)
        if self.use_batch_norm:
            x = self.batch_norm4(x)
        x = self.activation_funct(x)

        if self.set2set:
            x = self.set2set_pooling(x, batch, dim_size=10)
        else:
            x4a = global_max_pool(x, batch)
            x4b = global_mean_pool(x, batch)
            x = torch.cat([x4a, x4b], dim=1)

        # Linears:

        x = self.lin1(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin3(x)
        return x


class PNAnet6L(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct):
        super().__init__()

        self.activation_funct = activation_funct

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation", "linear"]

        self.conv1 = PNAConv(in_channels=num_node_features, out_channels=32,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        self.batch_norm1 = BatchNorm(32, momentum=0.05)
        self.set2set_pooling1 = utils.set2set_aggr(in_channels=32, processing_steps=1)

        self.conv2 = PNAConv(in_channels=32, out_channels=64,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        self.batch_norm2 = BatchNorm(64, momentum=0.05)
        self.set2set_pooling2 = utils.set2set_aggr(in_channels=64, processing_steps=1)

        self.conv3 = PNAConv(in_channels=64, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        self.batch_norm3 = BatchNorm(128, momentum=0.05)
        self.set2set_pooling3 = utils.set2set_aggr(in_channels=128, processing_steps=1)

        self.conv4 = PNAConv(in_channels=128, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        self.batch_norm4 = BatchNorm(128, momentum=0.05)
        self.set2set_pooling4 = utils.set2set_aggr(in_channels=128, processing_steps=1)

        self.conv5 = PNAConv(in_channels=128, out_channels=64,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        self.batch_norm5 = BatchNorm(64, momentum=0.05)
        self.set2set_pooling5 = utils.set2set_aggr(in_channels=64, processing_steps=1)

        self.conv6 = PNAConv(in_channels=64, out_channels=32,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=2, pre_layers=1, post_layers=1,
                             divide_input=True)
        # self.batch_norm6 = BatchNorm(32)
        self.set2set_pooling6 = utils.set2set_aggr(in_channels=32, processing_steps=1)

        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin1 = Linear((32*2 + 64*2 + 128*2 + 128*2 + 64*2 + 32*2), 896)
        self.batch_norm_l1 = BatchNorm(896, momentum=0.05)
        # self.lin1 = Linear((128 * 8), 896)
        # self.lin1 = Linear(((128 * 2) + 1), 128)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin2 = Linear(896, 384)
        self.dropout_lin3 = Dropout(p=dropout_lin_rest)
        # Introduce IntaRNA energy in a later layer, to give it more importance.
        self.lin3 = Linear(384, 64)  # + 1 if IntaRNA energy
        self.dropout_lin4 = Dropout(p=dropout_lin_rest)
        self.lin4 = Linear(64, 2)    # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2,
                dropout_conv_rest):
        # Convolutions:

        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv1(x, dropped_edge_index_1, dropped_edge_attr_1)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        #x1 = global_add_pool(x, batch)
        x1 = self.set2set_pooling1(x, batch, dim_size=10)

        dropped_edge_index_2, dropped_edge_attr_2 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv2(x, dropped_edge_index_2, dropped_edge_attr_2)
        #if self.training == True:
        #    x = x / conv_dropout  # To implement inverted dropout (so no scaling is necessary during testing)
        # -> not necessary, inverted dropout should be implemented automatically.
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        #x2 = global_add_pool(x, batch)
        x2 = self.set2set_pooling2(x, batch, dim_size=10)

        dropped_edge_index_3, dropped_edge_attr_3 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv3(x, dropped_edge_index_3, dropped_edge_attr_3)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        #x3 = global_add_pool(x, batch)
        x3 = self.set2set_pooling3(x, batch, dim_size=10)

        dropped_edge_index_4, dropped_edge_attr_4 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv4(x, dropped_edge_index_4, dropped_edge_attr_4)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        #x4 = global_add_pool(x, batch)
        x4 = self.set2set_pooling4(x, batch, dim_size=10)

        dropped_edge_index_5, dropped_edge_attr_5 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv5(x, dropped_edge_index_5, dropped_edge_attr_5)
        x = self.batch_norm5(x)
        x = self.activation_funct(x)

        #x5 = global_add_pool(x, batch)
        x5 = self.set2set_pooling5(x, batch, dim_size=10)

        dropped_edge_index_6, dropped_edge_attr_6 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_rest, force_undirected=True)
        x = self.conv6(x, dropped_edge_index_6, dropped_edge_attr_6)
        #x = self.batch_norm6(x)
        x = self.activation_funct(x)

        #x6a = global_add_pool(x, batch)
        #x6b = global_max_pool(x, batch)
        #x6c = global_mean_pool(x, batch)
        x6 = self.set2set_pooling6(x, batch, dim_size=10)

        # Linears:

        #en = intarna_energy.reshape(-1, 1)

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        if self.training:
            x = self.dropout_lin1(x)
        x = self.lin1(x)
        x = self.batch_norm_l1(x)
        x = self.activation_funct(x)
        if self.training:
            x = self.dropout_lin2(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training:
            x = self.dropout_lin3(x)
        x = self.lin3(x)
        x = self.activation_funct(x)
        if self.training:
            x = self.dropout_lin4(x)
        x = self.lin4(x)
        return x

class PNAnet4Lb(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct, rna_id=True):
        super().__init__()

        self.activation_funct = activation_funct

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.rna_id = rna_id

        if not self.rna_id:
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

        self.lin1 = Linear((128 * 2), 128)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin3 = Linear(64, 2)  # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2, dropout_conv_rest):
        # Convolutions:

        if not self.rna_id:
            x = x[:, :4]   # Don't use RNA ID attribute, only features for bases.

        edge_index_covalent = torch.stack((edge_index[0][covalent_edges], edge_index[1][covalent_edges]),
                                          dim=0)
        edge_index_non_covalent = torch.stack((edge_index[0][~covalent_edges], edge_index[1][~covalent_edges]),
                                              dim=0)  # ~: flips the boolean tensor

        edge_attr_covalent = edge_attr[covalent_edges]
        edge_attr_non_covalent = edge_attr[~covalent_edges]

        # Edge dropout with the same edges for all layers (not layer-wise):
        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index_non_covalent,
                                                                edge_attr=edge_attr_non_covalent,
                                                                p=dropout_conv_1_2, force_undirected=True)

        edge_index = torch.cat((edge_index_covalent, dropped_edge_index_1), dim=-1)
        edge_attr = torch.cat((edge_attr_covalent, dropped_edge_attr_1), dim=0)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin3(x)

        return x



class simp(torch.nn.Module):
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
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 2)  # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2, dropout_conv_rest):
        # Convolutions:

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)

        x = self.lin2(x)
        x = self.activation_funct(x)

        x = self.lin3(x)

        return x


# Create simple GCN-model!



# Create a modified version of PNAConv, with integrated Gated Graph Neural Network.
# Either:
# modify code of PNAConv class
# Or:
# Try to define a class like above and combine both in there.

from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree

from torch_geometric.nn.inits import reset

# for gated:
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.inits import uniform


class modifiedPNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers: (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0)
        #super(PNAConv, self).__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        # To add edge weights:?
        self.use_edge_weights = True

        # To add gated:?
        self.gated = True

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers
        if self.gated:
            self.F_in = self.out_channels

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        # To add gated:?
        self.num_layers = 1
        self.weight = Param(Tensor(self.num_layers, out_channels, out_channels))
        bias = True
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

        # to add gated:?
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if not self.gated:
            #print(edge_attr[:, :3])
            print(f"x.shape: {x.shape}, \nself.weight.shape: {self.weight.shape}\n")
            if self.divide_input:
                x = x.view(-1, self.towers, self.F_in)
            else:
                x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
            print(f"x.shape: {x.shape}\n")

            # propagate_type: (x: Tensor, edge_attr: OptTensor)
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

            out = torch.cat([x, out], dim=-1)
            outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
            out = torch.cat(outs, dim=1)
            print(f"out.shape: {out.shape}\n")

            x = self.lin(out)
            print(f"x.shape: {x.shape}\n \n")

        # to add gated:?
        if self.gated:

            # x.shape == [num-nodes, num-node, features]
            # self.weight.shape == [1, num-out-channels, num-out-channels]
            if x.size(-1) < self.out_channels:
                zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
                x = torch.cat([x, zero], dim=1)

            m = torch.matmul(x, self.weight[0])

            if self.divide_input:
                m = m.view(-1, self.towers, self.F_in)
            else:
                m = m.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

            # (x must be brought to the correct size (according to output channels))
            out = self.propagate(edge_index, x=m, edge_attr=edge_attr, size=None)
            out = torch.cat([m, out], dim=-1)
            outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
            out = torch.cat(outs, dim=1)
            x = self.rnn(out, x)

            # Additional self.lin(x) ?

        return x

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:
        if self.use_edge_weights:
            edge_weights = edge_attr[:, 3]
            edge_attr = edge_attr[:, :3]

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)

            if self.use_edge_weights:
                #print(f"x_1: {x_i.shape}, \nedge_weights: {edge_weights.shape}, \n "
                      #f"edge_attr: {edge_attr.shape}")
                # x_i.shape == x_j.shape == edge_attr.shape == [num-nodes, 1, num-node-features]
                # edge_weights.shape == [num-nodes]
                # edge_weights[:, None, None] == [num-nodes, 1, 1]
                x_i = edge_weights[:, None, None] * x_i
                x_j = edge_weights[:, None, None] * x_j
                edge_attr = edge_weights[:, None, None] * edge_attr
                # The node- and edge-features are weighted by the edge-weight before being concatenated.

            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')


class modifiedPNAnet(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct, rna_id=True):
        super().__init__()

        self.activation_funct = activation_funct
        self.rna_id = rna_id

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        if not self.rna_id:
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

        self.lin1 = Linear((128 * 2), 128)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin3 = Linear(64, 2)  # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2, dropout_conv_rest):

        if not self.rna_id:
            x = x[:, :4]   # Don't use RNA ID attribute, only features for bases.

        edge_index_covalent = torch.stack((edge_index[0][covalent_edges], edge_index[1][covalent_edges]),
                                          dim=0)
        edge_index_non_covalent = torch.stack((edge_index[0][~covalent_edges], edge_index[1][~covalent_edges]),
                                              dim=0)  # ~: flips the boolean tensor

        edge_attr_covalent = edge_attr[covalent_edges]
        edge_attr_non_covalent = edge_attr[~covalent_edges]

        #print(f"batch: {batch}\n"
        #      f"x.shape: {x.shape}\n \n")

        # Edge dropout with the same edges for all layers (not layer-wise):
        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index_non_covalent,
                                                                edge_attr=edge_attr_non_covalent,
                                                                p=dropout_conv_1_2, force_undirected=True)

        edge_index = torch.cat((edge_index_covalent, dropped_edge_index_1), dim=-1)
        edge_attr = torch.cat((edge_attr_covalent, dropped_edge_attr_1), dim=0)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        #print(f"x.shape: {x.shape}")

        x = self.lin1(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin3(x)

        #now = datetime.now()
        #print(f"after rest of forward: {now}")

        return x


#motif_lstm = torch.nn.LSTM(input_size=6, hidden_size=10, num_layers=2, bidirectional=True)

#D = 2  # because bidirectional, otherwise = 1
#out_channels = 10
#num_layers = 2
#h_0 = torch.randn(D * num_layers, batch_size, out_channels)
#c_0 = torch.randn(D * num_layers, batch_size, 10)

#batch_list = batch.to_data_list()

#from torch.nn.utils.rnn import pack_sequence
#input = pack_sequence([b.x for b in batch_list], enforce_sorted=False)

#output, (h_n, c_n) = motif_lstm(input, (h_0, c_0))

#motif_lstm()

from torch.nn.utils.rnn import pack_sequence

class moremodifiedPNAnet(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct, rna_id=True):
        super().__init__()

        self.activation_funct = activation_funct
        self.lstm_out_channels = 128
        self.lstm_num_layers = 2
        self.rna_id = rna_id

        if not self.rna_id:
            num_node_features = 4

        self.motif_lstm = torch.nn.LSTM(input_size=num_node_features, hidden_size=self.lstm_out_channels,
                                        num_layers=self.lstm_num_layers, bidirectional=True)

        self.D = 2  # because bidirectional, otherwise = 1

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

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

        lstm_feature_dims = self.D * self.lstm_num_layers * self.lstm_out_channels

        self.lin1 = Linear((128 * 2 + lstm_feature_dims), 256)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(256, 128)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin3 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin4 = Linear(64, 2)  # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, batch, dropout_conv_1_2, dropout_conv_rest, device):
        batch_size = batch.num_graphs

        h_0 = torch.randn(self.D * self.lstm_num_layers, batch_size, self.lstm_out_channels).to(device)
        c_0 = torch.randn(self.D * self.lstm_num_layers, batch_size, self.lstm_out_channels).to(device)
        batch_list = batch.to_data_list()
        if not self.rna_id:
            lstm_input = pack_sequence([b.x[:, :4] for b in batch_list], enforce_sorted=False)
        else:
            lstm_input = pack_sequence([b.x for b in batch_list], enforce_sorted=False)

        _, (h_n, c_n) = self.motif_lstm(lstm_input, (h_0, c_0))

        x = batch.x
        if not self.rna_id:
            x = x[:, :4]   # Don't use RNA ID attribute, only features for bases.

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        intarna_energy = batch.intarna_energy
        covalent_edges = batch.covalent_edges

        batch = batch.batch

        edge_index_covalent = torch.stack((edge_index[0][covalent_edges], edge_index[1][covalent_edges]),
                                          dim=0)
        edge_index_non_covalent = torch.stack((edge_index[0][~covalent_edges], edge_index[1][~covalent_edges]),
                                              dim=0)  # ~: flips the boolean tensor

        edge_attr_covalent = edge_attr[covalent_edges]
        edge_attr_non_covalent = edge_attr[~covalent_edges]

        #print(f"batch: {batch}\n"
        #      f"x.shape: {x.shape}\n \n")

        # Edge dropout with the same edges for all layers (not layer-wise):
        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index_non_covalent,
                                                                edge_attr=edge_attr_non_covalent,
                                                                p=dropout_conv_1_2, force_undirected=True)

        edge_index = torch.cat((edge_index_covalent, dropped_edge_index_1), dim=-1)
        edge_attr = torch.cat((edge_attr_covalent, dropped_edge_attr_1), dim=0)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = self.activation_funct(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        x = self.activation_funct(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        x = self.activation_funct(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.batch_norm4(x)
        x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        # introduce lstm-output

        x = torch.cat([x, h_n[0], h_n[1], h_n[2], h_n[3]], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin3(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin4(x)

        #now = datetime.now()
        #print(f"after rest of forward: {now}")

        return x


class sharedParamPNAnet(torch.nn.Module):
    def __init__(self, dropout_lin_1, dropout_lin_rest, deg, num_node_features, activation_funct):
        super().__init__()

        self.activation_funct = activation_funct

        self.num_conv_layers = 4

        self.use_batch_norm = True

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.conv1 = PNAConv(in_channels=num_node_features, out_channels=128,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)

        self.conv = PNAConv(in_channels=128, out_channels=128,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=4, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False)

        if self.use_batch_norm:
            self.batch_norm = BatchNorm(128)

        self.lin1 = Linear((128 * 2), 128)
        self.dropout_lin1 = Dropout(p=dropout_lin_1)
        self.lin2 = Linear(128, 64)
        self.dropout_lin2 = Dropout(p=dropout_lin_rest)
        self.lin3 = Linear(64, 2)  # 2 Output channels for CrossEntropyLoss (for BCELossWithLogits it would be 1)

    def forward(self, x, edge_index, edge_attr, intarna_energy, batch, covalent_edges, dropout_conv_1_2,
                dropout_conv_rest):

        dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                p=dropout_conv_1_2, force_undirected=True)
        x = self.conv1(x, dropped_edge_index_1, dropped_edge_attr_1)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.activation_funct(x)

        for layer in range(self.num_conv_layers - 1):
            dropped_edge_index_1, dropped_edge_attr_1 = dropout_adj(edge_index=edge_index, edge_attr=edge_attr,
                                                                    p=dropout_conv_1_2, force_undirected=True)
            x = self.conv(x, dropped_edge_index_1, dropped_edge_attr_1)
            #if self.use_batch_norm:
            #    x = self.batch_norm(x)
            x = self.activation_funct(x)

        x4a = global_max_pool(x, batch)
        x4b = global_mean_pool(x, batch)

        # Linears:

        x = torch.cat([x4a, x4b], dim=1)

        x = self.lin1(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin1(x)
        x = self.lin2(x)
        x = self.activation_funct(x)
        if self.training == True:
            x = self.dropout_lin2(x)
        x = self.lin3(x)
        return x


