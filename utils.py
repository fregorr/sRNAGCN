from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import torch
from typing import Optional
import pickle as rick

from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.utils import scatter


def custom_loader(dataset, batch_size, pos_class_weight=4):
    class_weights = [1, pos_class_weight]
    sample_weights = [0] * len(dataset)
    for idx, graph in enumerate(dataset):
        label = graph.y
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    return loader



class set2set_aggr(torch.nn.Module):
    r"""The Set2Set aggregation operator based on iterative content-based
    attention, as described in the `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
    """

    def __init__(self, in_channels: int, processing_steps: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm = torch.nn.LSTM(self.out_channels, in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'add') -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        assert index is not None
        return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def assert_index_present(self, index: Optional[Tensor]):
        # TODO Currently, not all aggregators support `ptr`. This assert helps
        # to ensure that we require `index` to be passed to the computation:
        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs "
                             f"(got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in "
                             f"first dimension (got '{dim}')")

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        h = (x.new_zeros((self.lstm.num_layers, dim_size, x.size(-1))),
             x.new_zeros((self.lstm.num_layers, dim_size, x.size(-1))))
        q_star = x.new_zeros(dim_size, self.out_channels)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(dim_size, self.in_channels)
            e = (x * q[index]).sum(dim=-1, keepdim=True)
            a = softmax(e, index, ptr, dim_size, dim)
            r = self.reduce(a * x, index, ptr, dim_size, dim, reduce='add')
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


# Just scratch code for getting subsets of the prepared dataset, for investigation, venn-diagramms...:
inspect_subsets = False
if inspect_subsets:
    file_interactions_overview_df = "data/test_ril_seq_dir_x_20_10_22/interactions_overview_df.pkl"

    with open(file_interactions_overview_df, "rb") as pickle_in:
        overview_df = rick.load(pickle_in)

    liste = list(overview_df[overview_df["source"] == "non_ril_seq"]["interaction_index"])
    string = ""
    for element in liste:
        e = str(element) + "\n"
        string = string + e

    out_file = "nvenn_non_ril_seq.txt"
    with open(out_file, "w") as f:
        f.write(string)

