# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.
#
# """File containing NASNet-series search space.
#
# The implementation is based on NDS.
# It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
# """
#
# from functools import partial
# from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast
# import torch.nn.functional as F
#
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal
#
# import torch
#
# import nni.nas.nn.pytorch as nn
# from nni.nas import model_wrapper
#
# from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
# from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
#
# from .utils.fixed import FixedFactory
# from .utils.pretrained import load_pretrained_weight
#
#
# # the following are NAS operations from
# # https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py
#
# OPS = {
#     'none': lambda C, stride, affine:Zero(stride),
#     'avg_pool_2x2': lambda C, stride, affine:nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
#     'avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'avg_pool_5x5': lambda C, stride, affine:nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
#     'max_pool_2x2': lambda C, stride, affine:nn.MaxPool2d(2, stride=stride, padding=0),
#     'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(3, stride=stride, padding=1),
#     'max_pool_5x5': lambda C, stride, affine:nn.MaxPool2d(5, stride=stride, padding=2),
#     'max_pool_7x7': lambda C, stride, affine:nn.MaxPool2d(7, stride=stride, padding=3),
#     'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C),
#     # 'skip_connect': lambda C, stride, affine:nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C, affine=affine),
#     'conv_1x1': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(C, affine=affine)),
#
#     'sep_conv_3x3': lambda C, stride, affine:PrivSepConv(C, C, 3, stride, 1, relu()),
#     'sep_conv_5x5': lambda C, stride, affine:PrivSepConv(C, C, 5, stride, 2, relu()),
#
#     'sep_conv_7x7': lambda C, stride, affine:SepConv(C, C, 7, stride, 3, affine=affine),
#
#     'dil_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
#     'dil_conv_5x5': lambda C, stride, affine:PrivDilConv(C, C, 5, stride, 4, 2, relu()),
#
#     'dil_sep_conv_3x3': lambda C, stride, affine:DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
#     'conv_3x1_1x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_7x1_1x7': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
# #############################################################################################################################################################################################
#     'priv_skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'priv_avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'priv_max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#     'priv_sep_conv_3x3_relu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, relu()),
#     'priv_sep_conv_3x3_elu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, elu()),
#     'priv_sep_conv_3x3_tanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, tanh()),
#     'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, sigmoid()),
#     'priv_sep_conv_3x3_selu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, selu()),
#     'priv_sep_conv_3x3_htanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, htanh()),
#     'priv_sep_conv_3x3_linear': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, Identity()),
# }
#
# class PrivSepConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super(PrivSepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
#                       bias=False, groups=C_out),
#             GN(C_out),
#             Act,
#         )
#
#     def forward(self, x):
#         x = self.op(x)
#         return x
#
#
# class PrivFactorizedReduce(nn.Module):
#     def __init__(self, C_in, C_out, Act=None):
#         super(PrivFactorizedReduce, self).__init__()
#         assert C_out % 2 == 0
#         self.relu = Act
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.bn = GN(C_out)
#
#     def forward(self, x):
#         if self.relu is not None:
#             x = self.relu(x)
#         if x.size(2)%2!=0:
#             x = F.pad(x, (1,0,1,0), "constant", 0)
#
#         out1 = self.conv_1(x)
#         out2 = self.conv_2(x[:, :, 1:, 1:])
#
#         out = torch.cat([out1, out2], dim=1)
#         out = self.bn(out)
#         return out
#
# class PrivDilConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
#         super(PrivDilConv, self).__init__()
#         self.op = nn.Sequential(
#             Act,
#             nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
#                       padding=padding, dilation=dilation, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out)
#         )
#
#     def forward(self, x):
#         return self.op(x)
#
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
# def GN(plane):
#     return nn.GroupNorm(4, plane, affine=False)
#
# def relu():
#     return nn.ReLU()
#
# def elu():
#     return nn.ELU()
#
# def tanh():
#     return nn.Tanh()
#
# def htanh():
#     return nn.Hardtanh()
#
# def sigmoid():
#     return nn.Sigmoid()
#
# def selu():
#     return nn.SELU()
#
# class ReLUConvBN(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_out, kernel_size, stride=stride,
#                 padding=padding, bias=False
#             ),
#             # nn.BatchNorm2d(C_out, affine=affine)
#             GN(C_out)
#         )
#
#
# class DilConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class SepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class DilSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class Zero(nn.Module):
#
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)
#
#
# class FactorizedReduce(nn.Module):
#
#     def __init__(self, C_in, C_out, affine=True):
#         super().__init__()
#         if isinstance(C_out, int):
#             assert C_out % 2 == 0
#         else:   # is a value choice
#             assert all(c % 2 == 0 for c in C_out.all_options())
#         self.relu = nn.ReLU(inplace=False)
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         # self.bn = nn.BatchNorm2d(C_out, affine=affine)
#         self.bn = GN(C_out)
#         self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
#
#     def forward(self, x):
#         x = self.relu(x)
#         y = self.pad(x)
#         out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
#         out = self.bn(out)
#         return out
#
#
# class DropPath_(nn.Module):
#     # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
#     def __init__(self, drop_prob=0.):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.training and self.drop_prob > 0.:
#             keep_prob = 1. - self.drop_prob
#             mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
#             return x.div(keep_prob).mul(mask)
#         return x
#
#
# class AuxiliaryHead(nn.Module):
#     def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
#         super().__init__()
#         if dataset == 'imagenet':
#             # assuming input size 14x14
#             stride = 2
#         elif dataset == 'cifar':
#             stride = 3
#
#         self.features = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
#             nn.Conv2d(C, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 768, 2, bias=False),
#             nn.BatchNorm2d(768),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(768, num_labels)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x
#
#
# class CellPreprocessor(nn.Module):
#     """
#     Aligning the shape of predecessors.
#     是一个用于细胞（cell）构建过程中对输入进行预处理的类。它主要用于对细胞的前驱节点进行形状对齐操作
#     If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
#     See :class:`CellBuilder` on how to calculate those channel numbers.
#     """
#
#     def __init__(self, C_pprev: nn.MaybeChoice[int], C_prev: nn.MaybeChoice[int], C: nn.MaybeChoice[int], last_cell_reduce: bool) -> None:
#         super().__init__()
#
#         if last_cell_reduce:
#             self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
#         else:
#             self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
#         self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)
#
#     def forward(self, cells):
#         assert len(cells) == 2
#         pprev, prev = cells
#         pprev = self.pre0(pprev)
#         prev = self.pre1(prev)
#
#         return [pprev, prev]
#
#
# class CellPostprocessor(nn.Module):
#     """
#     The cell outputs previous cell + this cell, so that cells can be directly chained.
#     """
#
#     def forward(self, this_cell, previous_cells):
#         return [previous_cells[-1], this_cell]
#
#
# class CellBuilder:
#     """The cell builder is used in Repeat.
#     Builds an cell each time it's "called".
#     Note that the builder is ephemeral, it can only be called once for every index.
#     """
#
#     def __init__(self, op_candidates: List[str],
#                  C_prev_in: nn.MaybeChoice[int],
#                  C_in: nn.MaybeChoice[int],
#                  C: nn.MaybeChoice[int],
#                  num_nodes: int,
#                  merge_op: Literal['all', 'loose_end'],
#                  first_cell_reduce: bool, last_cell_reduce: bool,
#                  drop_path_prob: float):
#         self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
#         self.C_in = C_in                # This is the out channesl of last cell.
#         self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
#         self.op_candidates = op_candidates
#         self.num_nodes = num_nodes
#         self.merge_op: Literal['all', 'loose_end'] = merge_op
#         self.first_cell_reduce = first_cell_reduce
#         self.last_cell_reduce = last_cell_reduce
#         self.drop_path_prob = drop_path_prob
#         self._expect_idx = 0
#
#         # It takes an index that is the index in the repeat.
#         # Number of predecessors for each cell is fixed to 2.
#         self.num_predecessors = 2
#
#         # Number of ops per node is fixed to 2.
#         self.num_ops_per_node = 2
#
#     def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,
#                    op: str, channels: int, is_reduction_cell: bool):
#         if is_reduction_cell and (
#             input_index is None or input_index < self.num_predecessors
#         ):  # could be none when constructing search space
#             stride = 2
#         else:
#             stride = 1
#         operation = OPS[op](channels, stride, True)
#         if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
#             # Omit drop-path when operation is skip connect.
#             # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
#             return nn.Sequential(operation, DropPath_(self.drop_path_prob))
#         return operation
#
#     #CellBuilder中的__call__方法用于实际构建cell。在该方法中，根据当前的repeat_idx确定是否为降采样的cell，并创建CellPreprocessor对象。
#     def __call__(self, repeat_idx: int):#在构建多个细胞的过程中，可以使用 repeat_idx 来跟踪当前正在构建的细胞是第几个重复的细胞
#         if self._expect_idx != repeat_idx:
#             raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')
#
#         # Reduction cell means stride = 2 and channel multiplied by 2.
#         is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce
#
#         # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
#         #该对象用于处理Cell的输入通道和输出通道的调整。
#         preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)
#         #用于存储操作的工厂函数。工厂函数根据指定的操作名称和其他参数来创建操作对象，其中包括使用self.op_factory方法创建的操作对象。
#         ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
#         for op in self.op_candidates:
#             ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)
#         #对象包含了一组操作节点，形成了一个完整的Cell。
#         cell = nn.Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
#                        preprocessor=preprocessor, postprocessor=CellPostprocessor(),
#                        label='reduce' if is_reduction_cell else 'normal')
#
#         # update state
#         self.C_prev_in = self.C_in
#         self.C_in = self.C * len(cell.output_node_indices)
#         self.last_cell_reduce = is_reduction_cell
#         self._expect_idx += 1
#         return cell
#
# #NDSStage包含多个Cell，通过继承nn.Repeat类来实现。nn.Repeat允许多次重复执行同一个操作，因此可以方便地构建多个Cell。
# class NDSStage(nn.Repeat):
#     """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.
#
#     In NDS, we can't simply use Repeat to stack the blocks,
#     because the output shape of each stacked block can be different.
#     This is a problem for one-shot strategy because they assume every possible candidate
#     should return values of the same shape.
#
#     Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
#     to manually align the shapes -- specifically, to transform the first block in each stage.
#
#     This is not required though, when depth is not changing, or the mutable depth causes no problem
#     (e.g., when the minimum depth is large enough).
#
#     .. attention::
#
#        Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
#        which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
#     """
#
#     estimated_out_channels_prev: int
#     """Output channels of cells in last stage.表示上一个阶段中的Cell的输出通道数"""
#
#     estimated_out_channels: int
#     """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``.表示该阶段中的Cell的输出通道数"""
#
#     downsampling: bool
#     """This stage has downsampling表示该阶段是否进行降采样"""
#
#     #用于创建一个变换模块，用于将第一个Cell的输出形状与该阶段中其他Cell的输出形状对齐。
#     def first_cell_transformation_factory(self) -> Optional[nn.Module]:
#         """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
#         if self.downsampling:
#             return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
#         elif self.estimated_out_channels_prev is not self.estimated_out_channels:
#             # Can't use != here, ValueChoice doesn't support
#             return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
#         return None
#
#
# class NDSStagePathSampling(PathSamplingRepeat):
#     """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in sampled or self.first_cell_transformation is None:
#             return super().reduction(items, sampled)
#         # items[0] must be the result of first cell
#         assert len(items[0]) == 2
#         # Only apply the transformation on "prev" output.
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, sampled)
#
#
# class NDSStageDifferentiable(DifferentiableMixedRepeat):
#     """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             # Only interesting when depth is mutable
#             softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice,
#                 softmax,
#                 memo
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(
#         self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in depths or self.first_cell_transformation is None:
#             return super().reduction(items, weights, depths)
#         # Same as NDSStagePathSampling
#         assert len(items[0]) == 2
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, weights, depths)
#
#
# _INIT_PARAMETER_DOCS = """
#
#     Notes
#     -----
#
#     To use NDS spaces with one-shot strategies,
#     especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
#     please use :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
#     and :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
#     This is because the output shape of each stacked block in :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStage` can be different.
#     For example::
#
#         from nni.retiarii.hub.pytorch.nasnet import NDSStageDifferentiable
#         darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])
#
#     Parameters
#     ----------
#     width
#         A fixed initial width or a tuple of widths to choose from.
#     num_cells
#         A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
#     dataset
#         The essential differences are in "stem" cells, i.e., how they process the raw image input.
#         Choosing "imagenet" means more downsampling at the beginning of the network.
#     auxiliary_loss
#         If true, another auxiliary classification head will produce the another prediction.
#         This makes the output of network two logits in the training phase.
#     drop_path_prob
#         Apply drop path. Enabled when it's set to be greater than 0.
#
# """
#
#
# class NDS(nn.Module):
#     __doc__ = """
#     The unified version of NASNet search space.
#
#     We follow the implementation in
#     `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
#     See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.
#
#     Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
#     ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.
#
#     NDS has a speciality that it has mutable depths/widths.
#     This is implemented by accepting a list of int as ``num_cells`` / ``width``.
#     """ + _INIT_PARAMETER_DOCS.rstrip() + """
#     op_candidates
#         List of operator candidates. Must be from ``OPS``.
#     merge_op
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     num_nodes_per_cell
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     """
#
#     def __init__(self,
#                  op_candidates: List[str],
#                  merge_op: Literal['all', 'loose_end'] = 'all',
#                  num_nodes_per_cell: int = 4,
#                  width: Union[Tuple[int, ...], int] = 16,
#                  num_cells: Union[Tuple[int, ...], int] = 20,#6
#                  dataset: Literal['cifar', 'imagenet'] = 'imagenet',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__()
#
#         self.dataset = dataset
#         self.num_labels = 10 if dataset == 'cifar' else 1000
#         self.auxiliary_loss = auxiliary_loss
#         self.drop_path_prob = drop_path_prob
#
#         # preprocess the specified width and depth
#         if isinstance(width, Iterable):
#             C = nn.ValueChoice(list(width), label='width')
#         else:
#             C = width
#
#         self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
#         if isinstance(num_cells, Iterable):
#             self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
#         #[2,2,2]每个阶段（stage）中重复构建的Cell的数量
#         num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]
#
#         # auxiliary head is different for network targetted at different datasets
#         if dataset == 'imagenet':
#             self.stem0 = nn.Sequential(
#                 nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(cast(int, C // 2)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             self.stem1 = nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             C_pprev = C_prev = C_curr = C
#             last_cell_reduce = True
#         elif dataset == 'cifar':
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
#                 #nn.BatchNorm2d(cast(int, 3 * C))
#                 GN(cast(int, 3 * C))
#             )
#             C_pprev = C_prev = 3 * C
#             C_curr = C
#             last_cell_reduce = False
#         else:
#             raise ValueError(f'Unsupported dataset: {dataset}')
#
#         self.stages = nn.ModuleList()
#         for stage_idx in range(3):
#             if stage_idx > 0:
#                 C_curr *= 2
#                 #"这部分就是reduction cell"
#             # For a stage, we get C_in, C_curr, and C_out.
#             # C_in is only used in the first cell.
#             # C_curr is number of channels for each operator in current stage.
#             # C_out is usually `C * num_nodes_per_cell` because of concat operator.
#             #NDSStage的构造函数接受一个CellBuilder对象和一个整数num_cells作为参数。CellBuilder用于构建单个Cell，num_cells表示在该阶段中要构建的Cell的数量。
#             cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
#                                        merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
#             stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])
#
#             if isinstance(stage, NDSStage):
#                 stage.estimated_out_channels_prev = cast(int, C_prev)
#                 stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
#                 stage.downsampling = stage_idx > 0
#
#             self.stages.append(stage)
#
#             # NOTE: output_node_indices will be computed on-the-fly in trial code.
#             # When constructing model space, it's just all the nodes in the cell,
#             # which happens to be the case of one-shot supernet.
#
#             # C_pprev is output channel number of last second cell among all the cells already built.
#             if len(stage) > 1:
#                 # Contains more than one cell
#                 C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
#             else:
#                 # Look up in the out channels of last stage.
#                 C_pprev = C_prev
#
#             # This was originally,
#             # C_prev = num_nodes_per_cell * C_curr.
#             # but due to loose end, it becomes,
#             C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr
#
#             # Useful in aligning the pprev and prev cell.
#             last_cell_reduce = cell_builder.last_cell_reduce
#
#             if stage_idx == 2:
#                 C_to_auxiliary = C_prev
#
#         if auxiliary_loss:
#             assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
#             self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore
#
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
#
#     def forward(self, inputs):
#         if self.dataset == 'imagenet':
#             s0 = self.stem0(inputs)
#             s1 = self.stem1(s0)
#         else:
#             s0 = s1 = self.stem(inputs)
#
#         for stage_idx, stage in enumerate(self.stages):
#             if stage_idx == 2 and self.auxiliary_loss and self.training:
#                 assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
#                 for block_idx, block in enumerate(stage):
#                     # auxiliary loss is attached to the first cell of the last stage.
#                     s0, s1 = block([s0, s1])
#                     if block_idx == 0:
#                         logits_aux = self.auxiliary_head(s1)
#             else:
#                 s0, s1 = stage([s0, s1])
#
#         out = self.global_pooling(s1)
#         print('==============='+out.size())
#         logits = self.classifier(out.view(out.size(0), -1))
#         if self.training and self.auxiliary_loss:
#             return logits, logits_aux  # type: ignore
#         else:
#             return logits
#
#     def set_drop_path_prob(self, drop_prob):
#         """
#         Set the drop probability of Drop-path in the network.
#         Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
#         """
#         for module in self.modules():
#             if isinstance(module, DropPath_):
#                 module.drop_prob = drop_prob
#
#     @classmethod
#     def fixed_arch(cls, arch: dict) -> FixedFactory:
#         return FixedFactory(cls, arch)
#
#
# @model_wrapper
# class NASNet(NDS):
#     __doc__ = """
#     Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~NASNet.NASNET_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     NASNET_OPS = [
#         'skip_connect',
#         'conv_3x1_1x3',
#         'conv_7x1_1x7',
#         'dil_conv_3x3',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'max_pool_5x5',
#         'max_pool_7x7',
#         'conv_1x1',
#         'conv_3x3',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.NASNET_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class ENAS(NDS):
#     __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~ENAS.ENAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     # ENAS_OPS = [
#     #     'skip_connect',
#     #     'sep_conv_3x3',
#     #     'sep_conv_5x5',
#     #     'avg_pool_3x3',
#     #     'max_pool_3x3',
#     # ]
#     ENAS_OPS = [
#         'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         # 'sep_conv_7x7',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
# #我们希望在子类中保留父类方法的行为，可以使用super()函数来调用父类的方法并在其基础上进行修改
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         print("=11111111111111=====111111========11111==")
#         super().__init__(self.ENAS_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class AmoebaNet(NDS):
#     __doc__ = """Search space proposed in
#     `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~AmoebaNet.AMOEBA_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     AMOEBA_OPS = [
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_sep_conv_3x3',
#         'conv_7x1_1x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#
#         super().__init__(self.AMOEBA_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class PNAS(NDS):
#     __doc__ = """Search space proposed in
#     `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
#     """ + _INIT_PARAMETER_DOCS
#
#     PNAS_OPS = [
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'conv_7x1_1x7',
#         'skip_connect',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_conv_3x3',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.PNAS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class DARTS(NDS):
#     __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
#     It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
#
#     .. note::
#
#         ``none`` is not included in the operator candidates.
#         It has already been handled in the differentiable implementation of cell.
#
#     """ + _INIT_PARAMETER_DOCS
#
#     DARTS_OPS = [
#         # 'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.DARTS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=4,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#     @classmethod
#     def load_searched_model(
#         cls, name: str,
#         pretrained: bool = False, download: bool = False, progress: bool = True
#     ) -> nn.Module:
#
#         init_kwargs = {}  # all default
#
#         if name == 'darts-v2':
#             init_kwargs.update(
#                 num_cells=20,
#                 width=36,
#             )
#             arch = {
#                 'normal/op_2_0': 'sep_conv_3x3',
#                 'normal/op_2_1': 'sep_conv_3x3',
#                 'normal/input_2_0': 0,
#                 'normal/input_2_1': 1,
#                 'normal/op_3_0': 'sep_conv_3x3',
#                 'normal/op_3_1': 'sep_conv_3x3',
#                 'normal/input_3_0': 0,
#                 'normal/input_3_1': 1,
#                 'normal/op_4_0': 'sep_conv_3x3',
#                 'normal/op_4_1': 'skip_connect',
#                 'normal/input_4_0': 1,
#                 'normal/input_4_1': 0,
#                 'normal/op_5_0': 'skip_connect',
#                 'normal/op_5_1': 'dil_conv_3x3',
#                 'normal/input_5_0': 0,
#                 'normal/input_5_1': 2,
#                 'reduce/op_2_0': 'max_pool_3x3',
#                 'reduce/op_2_1': 'max_pool_3x3',
#                 'reduce/input_2_0': 0,
#                 'reduce/input_2_1': 1,
#                 'reduce/op_3_0': 'skip_connect',
#                 'reduce/op_3_1': 'max_pool_3x3',
#                 'reduce/input_3_0': 2,
#                 'reduce/input_3_1': 1,
#                 'reduce/op_4_0': 'max_pool_3x3',
#                 'reduce/op_4_1': 'skip_connect',
#                 'reduce/input_4_0': 0,
#                 'reduce/input_4_1': 2,
#                 'reduce/op_5_0': 'skip_connect',
#                 'reduce/op_5_1': 'max_pool_3x3',
#                 'reduce/input_5_0': 2,
#                 'reduce/input_5_1': 1
#             }
#
#         else:
#             raise ValueError(f'Unsupported architecture with name: {name}')
#
#         model_factory = cls.fixed_arch(arch)
#         model = model_factory(**init_kwargs)
#
#         if pretrained:
#             weight_file = load_pretrained_weight(name, download=download, progress=progress)
#             pretrained_weights = torch.load(weight_file)
#             model.load_state_dict(pretrained_weights)
#
#         return model
# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.
#
# """File containing NASNet-series search space.
#
# The implementation is based on NDS.
# It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
# """
#
# from functools import partial
# from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast
# import torch.nn.functional as F
#
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal
#
# import torch
#
# import nni.nas.nn.pytorch as nn
# from nni.nas import model_wrapper
#
# from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
# from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
#
# from .utils.fixed import FixedFactory
# from .utils.pretrained import load_pretrained_weight
#
#
# # the following are NAS operations from
# # https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py
#
# OPS = {
#     'none': lambda C, stride, affine:Zero(stride),
#     'avg_pool_2x2': lambda C, stride, affine:nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
#     'avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'avg_pool_5x5': lambda C, stride, affine:nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
#     'max_pool_2x2': lambda C, stride, affine:nn.MaxPool2d(2, stride=stride, padding=0),
#     'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(3, stride=stride, padding=1),
#     'max_pool_5x5': lambda C, stride, affine:nn.MaxPool2d(5, stride=stride, padding=2),
#     'max_pool_7x7': lambda C, stride, affine:nn.MaxPool2d(7, stride=stride, padding=3),
#     'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C),
#     # 'skip_connect': lambda C, stride, affine:nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C, affine=affine),
#     'conv_1x1': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(C, affine=affine)),
#
#     'sep_conv_3x3': lambda C, stride, affine:PrivSepConv(C, C, 3, stride, 1, relu()),
#     'sep_conv_5x5': lambda C, stride, affine:PrivSepConv(C, C, 5, stride, 2, relu()),
#
#     # 'sep_conv_7x7': lambda C, stride, affine:SepConv(C, C, 7, stride, 3, affine=affine),
#     'sep_conv_7x7': lambda C, stride, affine: PrivSepConv(C, C, 7, stride, 3, relu()),
#     'dil_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
#     'dil_conv_5x5': lambda C, stride, affine:PrivDilConv(C, C, 5, stride, 4, 2, relu()),
#
#     'dil_sep_conv_3x3': lambda C, stride, affine:DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
#     # 'conv_3x1_1x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     # 'conv_7x1_1x7': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x1_1x3': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
#                                                             nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1),
#                                                                       bias=False),
#                                                             nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0),
#                                                                       bias=False), GN(C)),
#     'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
#                                                             nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3),
#                                                                       bias=False),
#                                                             nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0),
#                                                                       bias=False), GN(C)),
# #############################################################################################################################################################################################
#     'priv_skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'priv_avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'priv_max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#     'priv_sep_conv_3x3_relu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, relu()),
#     'priv_sep_conv_3x3_elu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, elu()),
#     'priv_sep_conv_3x3_tanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, tanh()),
#     'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, sigmoid()),
#     'priv_sep_conv_3x3_selu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, selu()),
#     'priv_sep_conv_3x3_htanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, htanh()),
#     'priv_sep_conv_3x3_linear': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, Identity()),
# }
#
# class PrivSepConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super(PrivSepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
#                       bias=False, groups=C_out),
#             GN(C_out),
#             Act,
#         )
#
#     def forward(self, x):
#         x = self.op(x)
#         return x
#
# class StdSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super().__init__(
#             Act,
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             GN(C_in),
#             Act,
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out),
#         )
#
# class PrivFactorizedReduce(nn.Module):
#     def __init__(self, C_in, C_out, Act=None):
#         super(PrivFactorizedReduce, self).__init__()
#         assert C_out % 2 == 0
#         self.relu = Act
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.bn = GN(C_out)
#
#     def forward(self, x):
#         if self.relu is not None:
#             x = self.relu(x)
#         if x.size(2)%2!=0:
#             x = F.pad(x, (1,0,1,0), "constant", 0)
#
#         out1 = self.conv_1(x)
#         out2 = self.conv_2(x[:, :, 1:, 1:])
#
#         out = torch.cat([out1, out2], dim=1)
#         out = self.bn(out)
#         return out
#
# class PrivDilConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
#         super(PrivDilConv, self).__init__()
#         self.op = nn.Sequential(
#             Act,
#             nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
#                       padding=padding, dilation=dilation, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out)
#         )
#
#     def forward(self, x):
#         return self.op(x)
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
# def GN(plane):
#     return nn.GroupNorm(8, plane, affine=False)
#
# def relu():
#     return nn.ReLU()
#
# def elu():
#     return nn.ELU()
#
# def tanh():
#     return nn.Tanh()
#
# def htanh():
#     return nn.Hardtanh()
#
# def sigmoid():
#     return nn.Sigmoid()
#
# def selu():
#     return nn.SELU()
#
# class ReLUConvBN(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_out, kernel_size, stride=stride,
#                 padding=padding, bias=False
#             ),
#             # nn.BatchNorm2d(C_out, affine=affine)
#             GN(C_out)
#         )
#
#
# class DilConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class SepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class DilSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             #nn.BatchNorm2d(C_in, affine=affine),
#             GN(C_in),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             #nn.BatchNorm2d(C_out, affine=affine),
#             GN(C_out),
#         )
#
#
# class Zero(nn.Module):
#
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)
#
#
# class FactorizedReduce(nn.Module):
#
#     def __init__(self, C_in, C_out, affine=True):
#         super().__init__()
#         if isinstance(C_out, int):
#             assert C_out % 2 == 0
#         else:   # is a value choice
#             assert all(c % 2 == 0 for c in C_out.all_options())
#         self.relu = nn.ReLU(inplace=False)
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         # self.bn = nn.BatchNorm2d(C_out, affine=affine)
#         self.bn = GN(C_out)
#         self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
#
#     def forward(self, x):
#         x = self.relu(x)
#         y = self.pad(x)
#         out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
#         out = self.bn(out)
#         return out
#
#
# class DropPath_(nn.Module):
#     # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
#     def __init__(self, drop_prob=0.):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.training and self.drop_prob > 0.:
#             keep_prob = 1. - self.drop_prob
#             mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
#             return x.div(keep_prob).mul(mask)
#         return x
#
#
# class AuxiliaryHead(nn.Module):
#     def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
#         super().__init__()
#         if dataset == 'imagenet':
#             # assuming input size 14x14
#             stride = 2
#         elif dataset == 'cifar':
#             stride = 3
#
#         self.features = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
#             nn.Conv2d(C, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 768, 2, bias=False),
#             nn.BatchNorm2d(768),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(768, num_labels)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x
#
#
# class CellPreprocessor(nn.Module):
#     """
#     Aligning the shape of predecessors.
#     是一个用于细胞（cell）构建过程中对输入进行预处理的类。它主要用于对细胞的前驱节点进行形状对齐操作
#     If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
#     See :class:`CellBuilder` on how to calculate those channel numbers.
#     """
#
#     def __init__(self, C_pprev: nn.MaybeChoice[int], C_prev: nn.MaybeChoice[int], C: nn.MaybeChoice[int], last_cell_reduce: bool) -> None:
#         super().__init__()
#
#         if last_cell_reduce:
#             self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
#         else:
#             self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
#         self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)
#
#     def forward(self, cells):
#         assert len(cells) == 2
#         pprev, prev = cells
#         pprev = self.pre0(pprev)
#         prev = self.pre1(prev)
#
#         return [pprev, prev]
#
#
# class CellPostprocessor(nn.Module):
#     """
#     The cell outputs previous cell + this cell, so that cells can be directly chained.
#     """
#
#     def forward(self, this_cell, previous_cells):
#         return [previous_cells[-1], this_cell]
#
#
# class CellBuilder:
#     """The cell builder is used in Repeat.
#     Builds an cell each time it's "called".
#     Note that the builder is ephemeral, it can only be called once for every index.
#     """
#
#     def __init__(self, op_candidates: List[str],
#                  C_prev_in: nn.MaybeChoice[int],
#                  C_in: nn.MaybeChoice[int],
#                  C: nn.MaybeChoice[int],
#                  num_nodes: int,
#                  merge_op: Literal['all', 'loose_end'],
#                  first_cell_reduce: bool, last_cell_reduce: bool,
#                  drop_path_prob: float):
#         self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
#         self.C_in = C_in                # This is the out channesl of last cell.
#         self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
#         self.op_candidates = op_candidates
#         self.num_nodes = num_nodes
#         self.merge_op: Literal['all', 'loose_end'] = merge_op
#         self.first_cell_reduce = first_cell_reduce
#         self.last_cell_reduce = last_cell_reduce
#         self.drop_path_prob = drop_path_prob
#         self._expect_idx = 0
#
#         # It takes an index that is the index in the repeat.
#         # Number of predecessors for each cell is fixed to 2.
#         self.num_predecessors = 2
#
#         # Number of ops per node is fixed to 2.
#         self.num_ops_per_node = 2
# #这个方法负责生成操作（operation）。在这个方法中，首先根据cell的类型（即是否为降采样cell）和输入节点的索引确定stride，然后根据指定的操作名和通道数创建operation。
#     def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,op: str, channels: int, is_reduction_cell: bool):
#         if is_reduction_cell and (input_index is None or input_index < self.num_predecessors):  # could be none when constructing search space
#             stride = 2
#         else:
#             stride = 1
#         operation = OPS[op](channels, stride, True)
#         if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
#             # Omit drop-path when operation is skip connect.
#             # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
#             return nn.Sequential(operation, DropPath_(self.drop_path_prob))
#         return operation
#
#     #CellBuilder中的__call__方法用于实际构建cell。在该方法中，根据当前的repeat_idx确定是否为降采样的cell，并创建CellPreprocessor对象。
#     def __call__(self, repeat_idx: int):#在构建多个细胞的过程中，可以使用 repeat_idx 来跟踪当前正在构建的细胞是第几个重复的细胞
#         if self._expect_idx != repeat_idx:
#             raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')
#
#         # Reduction cell means stride = 2 and channel multiplied by 2.判断是否为降采样cell。
#         is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce
#
#         # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
#         #创建一个CellPreprocessor对象，用于处理cell的输入和输出通道。
#         preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)
#         #用于存储操作的工厂函数。工厂函数根据指定的操作名称和其他参数来创建操作对象，其中包括使用self.op_factory方法创建的操作对象。
#         #创建一个字典ops_factory，存储所有可能的操作和对应的工厂函数。工厂函数根据操作名和其他参数创建operation。
#         ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
#         for op in self.op_candidates:
#             ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)
#         #对象包含了一组操作节点，形成了一个完整的Cell。
#         cell = nn.Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
#                        preprocessor=preprocessor, postprocessor=CellPostprocessor(),
#                        label='reduce' if is_reduction_cell else 'normal')
#
#         # update state
#         self.C_prev_in = self.C_in
#         self.C_in = self.C * len(cell.output_node_indices)
#         self.last_cell_reduce = is_reduction_cell
#         self._expect_idx += 1
#         return cell
#
# #NDSStage包含多个Cell，通过继承nn.Repeat类来实现。nn.Repeat允许多次重复执行同一个操作，因此可以方便地构建多个Cell。
# class NDSStage(nn.Repeat):
#     """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.
#
#     In NDS, we can't simply use Repeat to stack the blocks,
#     because the output shape of each stacked block can be different.
#     This is a problem for one-shot strategy because they assume every possible candidate
#     should return values of the same shape.
#
#     Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
#     to manually align the shapes -- specifically, to transform the first block in each stage.
#
#     This is not required though, when depth is not changing, or the mutable depth causes no problem
#     (e.g., when the minimum depth is large enough).
#
#     .. attention::
#
#        Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
#        which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
#     """
#
#     estimated_out_channels_prev: int
#     """Output channels of cells in last stage.表示上一个阶段中的Cell的输出通道数"""
#
#     estimated_out_channels: int
#     """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``.表示该阶段中的Cell的输出通道数"""
#
#     downsampling: bool
#     """This stage has downsampling表示该阶段是否进行降采样"""
#
#     #用于创建一个变换模块，用于将第一个Cell的输出形状与该阶段中其他Cell的输出形状对齐。
#     def first_cell_transformation_factory(self) -> Optional[nn.Module]:
#         """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
#         if self.downsampling:
#             return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
#         elif self.estimated_out_channels_prev is not self.estimated_out_channels:
#             # Can't use != here, ValueChoice doesn't support
#             return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
#         return None
#
#
# class NDSStagePathSampling(PathSamplingRepeat):
#     """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in sampled or self.first_cell_transformation is None:
#             return super().reduction(items, sampled)
#         # items[0] must be the result of first cell
#         assert len(items[0]) == 2
#         # Only apply the transformation on "prev" output.
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, sampled)
#
#
# class NDSStageDifferentiable(DifferentiableMixedRepeat):
#     """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             # Only interesting when depth is mutable
#             softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice,
#                 softmax,
#                 memo
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(
#         self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in depths or self.first_cell_transformation is None:
#             return super().reduction(items, weights, depths)
#         # Same as NDSStagePathSampling
#         assert len(items[0]) == 2
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, weights, depths)
#
#
# _INIT_PARAMETER_DOCS = """
#
#     Notes
#     -----
#
#     To use NDS spaces with one-shot strategies,
#     especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
#     please use :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
#     and :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
#     This is because the output shape of each stacked block in :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStage` can be different.
#     For example::
#
#         from nni.retiarii.hub.pytorch.nasnet import NDSStageDifferentiable
#         darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])
#
#     Parameters
#     ----------
#     width
#         A fixed initial width or a tuple of widths to choose from.
#     num_cells
#         A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
#     dataset
#         The essential differences are in "stem" cells, i.e., how they process the raw image input.
#         Choosing "imagenet" means more downsampling at the beginning of the network.
#     auxiliary_loss
#         If true, another auxiliary classification head will produce the another prediction.
#         This makes the output of network two logits in the training phase.
#     drop_path_prob
#         Apply drop path. Enabled when it's set to be greater than 0.
#
# """
#
#
# class NDS(nn.Module):
#     __doc__ = """
#     The unified version of NASNet search space.
#
#     We follow the implementation in
#     `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
#     See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.
#
#     Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
#     ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.
#
#     NDS has a speciality that it has mutable depths/widths.
#     This is implemented by accepting a list of int as ``num_cells`` / ``width``.
#     """ + _INIT_PARAMETER_DOCS.rstrip() + """
#     op_candidates
#         List of operator candidates. Must be from ``OPS``.
#     merge_op
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     num_nodes_per_cell
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     """
#
#     def __init__(self,
#                  op_candidates: List[str],
#                  merge_op: Literal['all', 'loose_end'] = 'all',
#                  num_nodes_per_cell: int = 4,
#                  width: Union[Tuple[int, ...], int] = 16,
#                  num_cells: Union[Tuple[int, ...], int] = 20,#6
#                  dataset: Literal['cifar', 'imagenet'] = 'imagenet',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__()
#
#         self.dataset = dataset
#         self.num_labels = 10 if dataset == 'cifar' else 1000
#         self.auxiliary_loss = auxiliary_loss
#         self.drop_path_prob = drop_path_prob
# #如果num_cells是一个列表，那么使用ValueChoice来表示这是一个候选值的集合，后续可以从中选择一个值。否则，直接使用num_cells作为网络的深度。
#         # preprocess the specified width and depth
#         if isinstance(width, Iterable):
#             C = nn.ValueChoice(list(width), label='width')
#         else:
#             C = width
#
#         self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
#         if isinstance(num_cells, Iterable):
#             self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
#         #[2,2,2]每个阶段（stage）中重复构建的Cell的数量
#         num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]
#
#         # auxiliary head is different for network targetted at different datasets
#         if dataset == 'imagenet':
#             self.stem0 = nn.Sequential(
#                 nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(cast(int, C // 2)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             self.stem1 = nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             C_pprev = C_prev = C_curr = C
#             last_cell_reduce = True
#         elif dataset == 'cifar':
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
#                 #nn.BatchNorm2d(cast(int, 3 * C))
#                 GN(cast(int, 3 * C))
#             )
#             C_pprev = C_prev = 3 * C
#             C_curr = C
#             last_cell_reduce = False
#         else:
#             raise ValueError(f'Unsupported dataset: {dataset}')
#
#         self.stages = nn.ModuleList()
#         #stage_idx > 0 作为是否对输入执行降采样（downsampling）的判断条件。stage_idx 为 0 代表网络的第一阶段，在这个阶段不执行降采样。而 stage_idx > 0 的情况（即第二阶段和第三阶段），在创建cell时会进行降采样。
#         for stage_idx in range(3):
#             if stage_idx > 0:
#                 C_curr *= 2
#             #NDSStage的构造函数接受一个CellBuilder对象和一个整数num_cells作为参数。CellBuilder用于构建单个Cell，num_cells表示在该阶段中要构建的Cell的数量。
#             cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
#             #在循环中，创建了三个阶段的cells。每个阶段的通道数C_curr可能会翻倍，对于第一个阶段，会创建一个CellBuilder对象，然后用这个CellBuilder对象创建一个NDSStage对象。这个NDSStage对象代表了一个阶段，其中包含了多个cell。它也记录了输入和输出的通道数，以及是否进行下采样。
#             stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])
#
#             if isinstance(stage, NDSStage):
#                 stage.estimated_out_channels_prev = cast(int, C_prev)
#                 stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
#                 stage.downsampling = stage_idx > 0
#
#             self.stages.append(stage)
# #在每个阶段的结束，更新了输入通道数（C_pprev和C_prev）以供下一个阶段使用，并设置了最后一个cell是否是下采样cell。
#             # NOTE: output_node_indices will be computed on-the-fly in trial code.
#             # When constructing model space, it's just all the nodes in the cell,
#             # which happens to be the case of one-shot supernet.
#
#             # C_pprev is output channel number of last second cell among all the cells already built.
#             if len(stage) > 1:
#                 # Contains more than one cell
#                 C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
#             else:
#                 # Look up in the out channels of last stage.
#                 C_pprev = C_prev
#
#             # This was originally,
#             # C_prev = num_nodes_per_cell * C_curr.
#             # but due to loose end, it becomes,
#             C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr
#
#             # Useful in aligning the pprev and prev cell.
#             last_cell_reduce = cell_builder.last_cell_reduce
#
#             if stage_idx == 2:
#                 C_to_auxiliary = C_prev
# #如果辅助损失被启用，那么在最后一个阶段创建了一个AuxiliaryHead。
#         if auxiliary_loss:
#             assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
#             self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore
#
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
#
#     def forward(self, inputs):
#         if self.dataset == 'imagenet':
#             s0 = self.stem0(inputs)
#             s1 = self.stem1(s0)
#         else:
#             s0 = s1 = self.stem(inputs)
# #这段代码实现了在网络的第三个阶段使用辅助损失来提高网络的训练效果。辅助损失是一种常用的技术，用于在网络的深层添加额外的损失函数，从而减少梯度消失问题，提高网络的性能。
#         for stage_idx, stage in enumerate(self.stages):
#             if stage_idx == 2 and self.auxiliary_loss and self.training:
#                 assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
#                 for block_idx, block in enumerate(stage):
#                     # auxiliary loss is attached to the first cell of the last stage.
#                     s0, s1 = block([s0, s1])
#                     if block_idx == 0:
#                         logits_aux = self.auxiliary_head(s1)
#             else:
#                 s0, s1 = stage([s0, s1])
#
#         out = self.global_pooling(s1)
#
#         logits = self.classifier(out.view(out.size(0), -1))
#         if self.training and self.auxiliary_loss:
#             return logits, logits_aux  # type: ignore
#         else:
#             return logits
#
#     def set_drop_path_prob(self, drop_prob):
#         """
#         Set the drop probability of Drop-path in the network.
#         Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
#         """
#         for module in self.modules():
#             if isinstance(module, DropPath_):
#                 module.drop_prob = drop_prob
#
#     @classmethod
#     def fixed_arch(cls, arch: dict) -> FixedFactory:
#         return FixedFactory(cls, arch)
#
#
# @model_wrapper
# class NASNet(NDS):
#     __doc__ = """
#     Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~NASNet.NASNET_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     NASNET_OPS = [
#         'skip_connect',
#         'conv_3x1_1x3',
#         'conv_7x1_1x7',
#         'dil_conv_3x3',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'max_pool_5x5',
#         'max_pool_7x7',
#         'conv_1x1',
#         'conv_3x3',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.NASNET_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class ENAS(NDS):
#     __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~ENAS.ENAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     # ENAS_OPS = [
#     #     'skip_connect',
#     #     'sep_conv_3x3',
#     #     'sep_conv_5x5',
#     #     'avg_pool_3x3',
#     #     'max_pool_3x3',
#     # ]
#     ENAS_OPS = [
#         'none',
#         'priv_max_pool_3x3',
#         'priv_avg_pool_3x3',
#         'priv_skip_connect',
#         'priv_sep_conv_3x3_relu',
#         'priv_sep_conv_3x3_selu',
#         'priv_sep_conv_3x3_tanh',
#         'priv_sep_conv_3x3_linear',
#         'priv_sep_conv_3x3_htanh',
#         'priv_sep_conv_3x3_sigmoid',
#     ]
#     """The candidate operations."""
# #我们希望在子类中保留父类方法的行为，可以使用super()函数来调用父类的方法并在其基础上进行修改
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         #在初始化过程中，它调用了父类NDS的__init__方法，并传递了一些参数，用于构建基本的模型架构搜索空间。
#         super().__init__(self.ENAS_OPS,
#                          merge_op='all',#原本是loose_end
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class AmoebaNet(NDS):
#     __doc__ = """Search space proposed in
#     `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~AmoebaNet.AMOEBA_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     AMOEBA_OPS = [
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_sep_conv_3x3',
#         'conv_7x1_1x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#
#         super().__init__(self.AMOEBA_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class PNAS(NDS):
#     __doc__ = """Search space proposed in
#     `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
#     """ + _INIT_PARAMETER_DOCS
#
#     PNAS_OPS = [
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'conv_7x1_1x7',
#         'skip_connect',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_conv_3x3',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.PNAS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class DARTS(NDS):
#     __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
#     It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
#
#     .. note::
#
#         ``none`` is not included in the operator candidates.
#         It has already been handled in the differentiable implementation of cell.
#
#     """ + _INIT_PARAMETER_DOCS
#
#     DARTS_OPS = [
#         # 'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.DARTS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=4,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#     @classmethod
#     def load_searched_model(
#         cls, name: str,
#         pretrained: bool = False, download: bool = False, progress: bool = True
#     ) -> nn.Module:
#
#         init_kwargs = {}  # all default
#
#         if name == 'darts-v2':
#             init_kwargs.update(
#                 num_cells=20,
#                 width=36,
#             )
#             arch = {
#                 'normal/op_2_0': 'sep_conv_3x3',
#                 'normal/op_2_1': 'sep_conv_3x3',
#                 'normal/input_2_0': 0,
#                 'normal/input_2_1': 1,
#                 'normal/op_3_0': 'sep_conv_3x3',
#                 'normal/op_3_1': 'sep_conv_3x3',
#                 'normal/input_3_0': 0,
#                 'normal/input_3_1': 1,
#                 'normal/op_4_0': 'sep_conv_3x3',
#                 'normal/op_4_1': 'skip_connect',
#                 'normal/input_4_0': 1,
#                 'normal/input_4_1': 0,
#                 'normal/op_5_0': 'skip_connect',
#                 'normal/op_5_1': 'dil_conv_3x3',
#                 'normal/input_5_0': 0,
#                 'normal/input_5_1': 2,
#                 'reduce/op_2_0': 'max_pool_3x3',
#                 'reduce/op_2_1': 'max_pool_3x3',
#                 'reduce/input_2_0': 0,
#                 'reduce/input_2_1': 1,
#                 'reduce/op_3_0': 'skip_connect',
#                 'reduce/op_3_1': 'max_pool_3x3',
#                 'reduce/input_3_0': 2,
#                 'reduce/input_3_1': 1,
#                 'reduce/op_4_0': 'max_pool_3x3',
#                 'reduce/op_4_1': 'skip_connect',
#                 'reduce/input_4_0': 0,
#                 'reduce/input_4_1': 2,
#                 'reduce/op_5_0': 'skip_connect',
#                 'reduce/op_5_1': 'max_pool_3x3',
#                 'reduce/input_5_0': 2,
#                 'reduce/input_5_1': 1
#             }
#
#         else:
#             raise ValueError(f'Unsupported architecture with name: {name}')
#
#         model_factory = cls.fixed_arch(arch)
#         model = model_factory(**init_kwargs)
#
#         if pretrained:
#             weight_file = load_pretrained_weight(name, download=download, progress=progress)
#             pretrained_weights = torch.load(weight_file)
#             model.load_state_dict(pretrained_weights)
#
#         return model


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File containing NASNet-series search space.

The implementation is based on NDS.
It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
"""

# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.
#
# """File containing NASNet-series search space.
#
# The implementation is based on NDS.
# It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
# """
#
# from functools import partial
# from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast
# import torch.nn.functional as F
#
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal
#
# import torch
#
# import nni.nas.nn.pytorch as nn
# from nni.nas import model_wrapper
#
# from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
# from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
#
# from .utils.fixed import FixedFactory
# from .utils.pretrained import load_pretrained_weight
#
#
# # the following are NAS operations from
# # https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py
#
# OPS = {
#     'none': lambda C, stride, affine:Zero(stride),
#     'avg_pool_2x2': lambda C, stride, affine:nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
#     'avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'avg_pool_5x5': lambda C, stride, affine:nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
#     'max_pool_2x2': lambda C, stride, affine:nn.MaxPool2d(2, stride=stride, padding=0),
#     'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(3, stride=stride, padding=1),
#     'max_pool_5x5': lambda C, stride, affine:nn.MaxPool2d(5, stride=stride, padding=2),
#     'max_pool_7x7': lambda C, stride, affine:nn.MaxPool2d(7, stride=stride, padding=3),
#     'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C),
#     # 'skip_connect': lambda C, stride, affine:nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C, affine=affine),
#     'conv_1x1': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(C, affine=affine)),
#
#     'sep_conv_3x3': lambda C, stride, affine:PrivSepConv(C, C, 3, stride, 1, relu()),
#     'sep_conv_5x5': lambda C, stride, affine:PrivSepConv(C, C, 5, stride, 2, relu()),
#
#     'sep_conv_7x7': lambda C, stride, affine:SepConv(C, C, 7, stride, 3, affine=affine),
#
#     'dil_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
#     'dil_conv_5x5': lambda C, stride, affine:PrivDilConv(C, C, 5, stride, 4, 2, relu()),
#
#     'dil_sep_conv_3x3': lambda C, stride, affine:DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
#     'conv_3x1_1x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_7x1_1x7': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
# #############################################################################################################################################################################################
#     'priv_skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'priv_avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'priv_max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#     'priv_sep_conv_3x3_relu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, relu()),
#     'priv_sep_conv_3x3_elu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, elu()),
#     'priv_sep_conv_3x3_tanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, tanh()),
#     'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, sigmoid()),
#     'priv_sep_conv_3x3_selu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, selu()),
#     'priv_sep_conv_3x3_htanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, htanh()),
#     'priv_sep_conv_3x3_linear': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, Identity()),
# }
#
# class PrivSepConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super(PrivSepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
#                       bias=False, groups=C_out),
#             GN(C_out),
#             Act,
#         )
#
#     def forward(self, x):
#         x = self.op(x)
#         return x
#
#
# class PrivFactorizedReduce(nn.Module):
#     def __init__(self, C_in, C_out, Act=None):
#         super(PrivFactorizedReduce, self).__init__()
#         assert C_out % 2 == 0
#         self.relu = Act
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.bn = GN(C_out)
#
#     def forward(self, x):
#         if self.relu is not None:
#             x = self.relu(x)
#         if x.size(2)%2!=0:
#             x = F.pad(x, (1,0,1,0), "constant", 0)
#
#         out1 = self.conv_1(x)
#         out2 = self.conv_2(x[:, :, 1:, 1:])
#
#         out = torch.cat([out1, out2], dim=1)
#         out = self.bn(out)
#         return out
#
# class PrivDilConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
#         super(PrivDilConv, self).__init__()
#         self.op = nn.Sequential(
#             Act,
#             nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
#                       padding=padding, dilation=dilation, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out)
#         )
#
#     def forward(self, x):
#         return self.op(x)
#
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
# def GN(plane):
#     return nn.GroupNorm(4, plane, affine=False)
#
# def relu():
#     return nn.ReLU()
#
# def elu():
#     return nn.ELU()
#
# def tanh():
#     return nn.Tanh()
#
# def htanh():
#     return nn.Hardtanh()
#
# def sigmoid():
#     return nn.Sigmoid()
#
# def selu():
#     return nn.SELU()
#
# class ReLUConvBN(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_out, kernel_size, stride=stride,
#                 padding=padding, bias=False
#             ),
#             # nn.BatchNorm2d(C_out, affine=affine)
#             GN(C_out)
#         )
#
#
# class DilConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class SepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class DilSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class Zero(nn.Module):
#
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)
#
#
# class FactorizedReduce(nn.Module):
#
#     def __init__(self, C_in, C_out, affine=True):
#         super().__init__()
#         if isinstance(C_out, int):
#             assert C_out % 2 == 0
#         else:   # is a value choice
#             assert all(c % 2 == 0 for c in C_out.all_options())
#         self.relu = nn.ReLU(inplace=False)
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         # self.bn = nn.BatchNorm2d(C_out, affine=affine)
#         self.bn = GN(C_out)
#         self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
#
#     def forward(self, x):
#         x = self.relu(x)
#         y = self.pad(x)
#         out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
#         out = self.bn(out)
#         return out
#
#
# class DropPath_(nn.Module):
#     # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
#     def __init__(self, drop_prob=0.):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.training and self.drop_prob > 0.:
#             keep_prob = 1. - self.drop_prob
#             mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
#             return x.div(keep_prob).mul(mask)
#         return x
#
#
# class AuxiliaryHead(nn.Module):
#     def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
#         super().__init__()
#         if dataset == 'imagenet':
#             # assuming input size 14x14
#             stride = 2
#         elif dataset == 'cifar':
#             stride = 3
#
#         self.features = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
#             nn.Conv2d(C, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 768, 2, bias=False),
#             nn.BatchNorm2d(768),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(768, num_labels)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x
#
#
# class CellPreprocessor(nn.Module):
#     """
#     Aligning the shape of predecessors.
#     是一个用于细胞（cell）构建过程中对输入进行预处理的类。它主要用于对细胞的前驱节点进行形状对齐操作
#     If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
#     See :class:`CellBuilder` on how to calculate those channel numbers.
#     """
#
#     def __init__(self, C_pprev: nn.MaybeChoice[int], C_prev: nn.MaybeChoice[int], C: nn.MaybeChoice[int], last_cell_reduce: bool) -> None:
#         super().__init__()
#
#         if last_cell_reduce:
#             self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
#         else:
#             self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
#         self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)
#
#     def forward(self, cells):
#         assert len(cells) == 2
#         pprev, prev = cells
#         pprev = self.pre0(pprev)
#         prev = self.pre1(prev)
#
#         return [pprev, prev]
#
#
# class CellPostprocessor(nn.Module):
#     """
#     The cell outputs previous cell + this cell, so that cells can be directly chained.
#     """
#
#     def forward(self, this_cell, previous_cells):
#         return [previous_cells[-1], this_cell]
#
#
# class CellBuilder:
#     """The cell builder is used in Repeat.
#     Builds an cell each time it's "called".
#     Note that the builder is ephemeral, it can only be called once for every index.
#     """
#
#     def __init__(self, op_candidates: List[str],
#                  C_prev_in: nn.MaybeChoice[int],
#                  C_in: nn.MaybeChoice[int],
#                  C: nn.MaybeChoice[int],
#                  num_nodes: int,
#                  merge_op: Literal['all', 'loose_end'],
#                  first_cell_reduce: bool, last_cell_reduce: bool,
#                  drop_path_prob: float):
#         self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
#         self.C_in = C_in                # This is the out channesl of last cell.
#         self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
#         self.op_candidates = op_candidates
#         self.num_nodes = num_nodes
#         self.merge_op: Literal['all', 'loose_end'] = merge_op
#         self.first_cell_reduce = first_cell_reduce
#         self.last_cell_reduce = last_cell_reduce
#         self.drop_path_prob = drop_path_prob
#         self._expect_idx = 0
#
#         # It takes an index that is the index in the repeat.
#         # Number of predecessors for each cell is fixed to 2.
#         self.num_predecessors = 2
#
#         # Number of ops per node is fixed to 2.
#         self.num_ops_per_node = 2
#
#     def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,
#                    op: str, channels: int, is_reduction_cell: bool):
#         if is_reduction_cell and (
#             input_index is None or input_index < self.num_predecessors
#         ):  # could be none when constructing search space
#             stride = 2
#         else:
#             stride = 1
#         operation = OPS[op](channels, stride, True)
#         if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
#             # Omit drop-path when operation is skip connect.
#             # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
#             return nn.Sequential(operation, DropPath_(self.drop_path_prob))
#         return operation
#
#     #CellBuilder中的__call__方法用于实际构建cell。在该方法中，根据当前的repeat_idx确定是否为降采样的cell，并创建CellPreprocessor对象。
#     def __call__(self, repeat_idx: int):#在构建多个细胞的过程中，可以使用 repeat_idx 来跟踪当前正在构建的细胞是第几个重复的细胞
#         if self._expect_idx != repeat_idx:
#             raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')
#
#         # Reduction cell means stride = 2 and channel multiplied by 2.
#         is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce
#
#         # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
#         #该对象用于处理Cell的输入通道和输出通道的调整。
#         preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)
#         #用于存储操作的工厂函数。工厂函数根据指定的操作名称和其他参数来创建操作对象，其中包括使用self.op_factory方法创建的操作对象。
#         ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
#         for op in self.op_candidates:
#             ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)
#         #对象包含了一组操作节点，形成了一个完整的Cell。
#         cell = nn.Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
#                        preprocessor=preprocessor, postprocessor=CellPostprocessor(),
#                        label='reduce' if is_reduction_cell else 'normal')
#
#         # update state
#         self.C_prev_in = self.C_in
#         self.C_in = self.C * len(cell.output_node_indices)
#         self.last_cell_reduce = is_reduction_cell
#         self._expect_idx += 1
#         return cell
#
# #NDSStage包含多个Cell，通过继承nn.Repeat类来实现。nn.Repeat允许多次重复执行同一个操作，因此可以方便地构建多个Cell。
# class NDSStage(nn.Repeat):
#     """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.
#
#     In NDS, we can't simply use Repeat to stack the blocks,
#     because the output shape of each stacked block can be different.
#     This is a problem for one-shot strategy because they assume every possible candidate
#     should return values of the same shape.
#
#     Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
#     to manually align the shapes -- specifically, to transform the first block in each stage.
#
#     This is not required though, when depth is not changing, or the mutable depth causes no problem
#     (e.g., when the minimum depth is large enough).
#
#     .. attention::
#
#        Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
#        which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
#     """
#
#     estimated_out_channels_prev: int
#     """Output channels of cells in last stage.表示上一个阶段中的Cell的输出通道数"""
#
#     estimated_out_channels: int
#     """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``.表示该阶段中的Cell的输出通道数"""
#
#     downsampling: bool
#     """This stage has downsampling表示该阶段是否进行降采样"""
#
#     #用于创建一个变换模块，用于将第一个Cell的输出形状与该阶段中其他Cell的输出形状对齐。
#     def first_cell_transformation_factory(self) -> Optional[nn.Module]:
#         """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
#         if self.downsampling:
#             return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
#         elif self.estimated_out_channels_prev is not self.estimated_out_channels:
#             # Can't use != here, ValueChoice doesn't support
#             return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
#         return None
#
#
# class NDSStagePathSampling(PathSamplingRepeat):
#     """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in sampled or self.first_cell_transformation is None:
#             return super().reduction(items, sampled)
#         # items[0] must be the result of first cell
#         assert len(items[0]) == 2
#         # Only apply the transformation on "prev" output.
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, sampled)
#
#
# class NDSStageDifferentiable(DifferentiableMixedRepeat):
#     """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             # Only interesting when depth is mutable
#             softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice,
#                 softmax,
#                 memo
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(
#         self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in depths or self.first_cell_transformation is None:
#             return super().reduction(items, weights, depths)
#         # Same as NDSStagePathSampling
#         assert len(items[0]) == 2
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, weights, depths)
#
#
# _INIT_PARAMETER_DOCS = """
#
#     Notes
#     -----
#
#     To use NDS spaces with one-shot strategies,
#     especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
#     please use :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
#     and :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
#     This is because the output shape of each stacked block in :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStage` can be different.
#     For example::
#
#         from nni.retiarii.hub.pytorch.nasnet import NDSStageDifferentiable
#         darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])
#
#     Parameters
#     ----------
#     width
#         A fixed initial width or a tuple of widths to choose from.
#     num_cells
#         A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
#     dataset
#         The essential differences are in "stem" cells, i.e., how they process the raw image input.
#         Choosing "imagenet" means more downsampling at the beginning of the network.
#     auxiliary_loss
#         If true, another auxiliary classification head will produce the another prediction.
#         This makes the output of network two logits in the training phase.
#     drop_path_prob
#         Apply drop path. Enabled when it's set to be greater than 0.
#
# """
#
#
# class NDS(nn.Module):
#     __doc__ = """
#     The unified version of NASNet search space.
#
#     We follow the implementation in
#     `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
#     See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.
#
#     Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
#     ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.
#
#     NDS has a speciality that it has mutable depths/widths.
#     This is implemented by accepting a list of int as ``num_cells`` / ``width``.
#     """ + _INIT_PARAMETER_DOCS.rstrip() + """
#     op_candidates
#         List of operator candidates. Must be from ``OPS``.
#     merge_op
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     num_nodes_per_cell
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     """
#
#     def __init__(self,
#                  op_candidates: List[str],
#                  merge_op: Literal['all', 'loose_end'] = 'all',
#                  num_nodes_per_cell: int = 4,
#                  width: Union[Tuple[int, ...], int] = 16,
#                  num_cells: Union[Tuple[int, ...], int] = 20,#6
#                  dataset: Literal['cifar', 'imagenet'] = 'imagenet',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__()
#
#         self.dataset = dataset
#         self.num_labels = 10 if dataset == 'cifar' else 1000
#         self.auxiliary_loss = auxiliary_loss
#         self.drop_path_prob = drop_path_prob
#
#         # preprocess the specified width and depth
#         if isinstance(width, Iterable):
#             C = nn.ValueChoice(list(width), label='width')
#         else:
#             C = width
#
#         self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
#         if isinstance(num_cells, Iterable):
#             self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
#         #[2,2,2]每个阶段（stage）中重复构建的Cell的数量
#         num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]
#
#         # auxiliary head is different for network targetted at different datasets
#         if dataset == 'imagenet':
#             self.stem0 = nn.Sequential(
#                 nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(cast(int, C // 2)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             self.stem1 = nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             C_pprev = C_prev = C_curr = C
#             last_cell_reduce = True
#         elif dataset == 'cifar':
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
#                 #nn.BatchNorm2d(cast(int, 3 * C))
#                 GN(cast(int, 3 * C))
#             )
#             C_pprev = C_prev = 3 * C
#             C_curr = C
#             last_cell_reduce = False
#         else:
#             raise ValueError(f'Unsupported dataset: {dataset}')
#
#         self.stages = nn.ModuleList()
#         for stage_idx in range(3):
#             if stage_idx > 0:
#                 C_curr *= 2
#                 #"这部分就是reduction cell"
#             # For a stage, we get C_in, C_curr, and C_out.
#             # C_in is only used in the first cell.
#             # C_curr is number of channels for each operator in current stage.
#             # C_out is usually `C * num_nodes_per_cell` because of concat operator.
#             #NDSStage的构造函数接受一个CellBuilder对象和一个整数num_cells作为参数。CellBuilder用于构建单个Cell，num_cells表示在该阶段中要构建的Cell的数量。
#             cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
#                                        merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
#             stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])
#
#             if isinstance(stage, NDSStage):
#                 stage.estimated_out_channels_prev = cast(int, C_prev)
#                 stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
#                 stage.downsampling = stage_idx > 0
#
#             self.stages.append(stage)
#
#             # NOTE: output_node_indices will be computed on-the-fly in trial code.
#             # When constructing model space, it's just all the nodes in the cell,
#             # which happens to be the case of one-shot supernet.
#
#             # C_pprev is output channel number of last second cell among all the cells already built.
#             if len(stage) > 1:
#                 # Contains more than one cell
#                 C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
#             else:
#                 # Look up in the out channels of last stage.
#                 C_pprev = C_prev
#
#             # This was originally,
#             # C_prev = num_nodes_per_cell * C_curr.
#             # but due to loose end, it becomes,
#             C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr
#
#             # Useful in aligning the pprev and prev cell.
#             last_cell_reduce = cell_builder.last_cell_reduce
#
#             if stage_idx == 2:
#                 C_to_auxiliary = C_prev
#
#         if auxiliary_loss:
#             assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
#             self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore
#
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
#
#     def forward(self, inputs):
#         if self.dataset == 'imagenet':
#             s0 = self.stem0(inputs)
#             s1 = self.stem1(s0)
#         else:
#             s0 = s1 = self.stem(inputs)
#
#         for stage_idx, stage in enumerate(self.stages):
#             if stage_idx == 2 and self.auxiliary_loss and self.training:
#                 assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
#                 for block_idx, block in enumerate(stage):
#                     # auxiliary loss is attached to the first cell of the last stage.
#                     s0, s1 = block([s0, s1])
#                     if block_idx == 0:
#                         logits_aux = self.auxiliary_head(s1)
#             else:
#                 s0, s1 = stage([s0, s1])
#
#         out = self.global_pooling(s1)
#         print('==============='+out.size())
#         logits = self.classifier(out.view(out.size(0), -1))
#         if self.training and self.auxiliary_loss:
#             return logits, logits_aux  # type: ignore
#         else:
#             return logits
#
#     def set_drop_path_prob(self, drop_prob):
#         """
#         Set the drop probability of Drop-path in the network.
#         Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
#         """
#         for module in self.modules():
#             if isinstance(module, DropPath_):
#                 module.drop_prob = drop_prob
#
#     @classmethod
#     def fixed_arch(cls, arch: dict) -> FixedFactory:
#         return FixedFactory(cls, arch)
#
#
# @model_wrapper
# class NASNet(NDS):
#     __doc__ = """
#     Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~NASNet.NASNET_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     NASNET_OPS = [
#         'skip_connect',
#         'conv_3x1_1x3',
#         'conv_7x1_1x7',
#         'dil_conv_3x3',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'max_pool_5x5',
#         'max_pool_7x7',
#         'conv_1x1',
#         'conv_3x3',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.NASNET_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class ENAS(NDS):
#     __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~ENAS.ENAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     # ENAS_OPS = [
#     #     'skip_connect',
#     #     'sep_conv_3x3',
#     #     'sep_conv_5x5',
#     #     'avg_pool_3x3',
#     #     'max_pool_3x3',
#     # ]
#     ENAS_OPS = [
#         'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         # 'sep_conv_7x7',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
# #我们希望在子类中保留父类方法的行为，可以使用super()函数来调用父类的方法并在其基础上进行修改
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         print("=11111111111111=====111111========11111==")
#         super().__init__(self.ENAS_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class AmoebaNet(NDS):
#     __doc__ = """Search space proposed in
#     `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~AmoebaNet.AMOEBA_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     AMOEBA_OPS = [
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_sep_conv_3x3',
#         'conv_7x1_1x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#
#         super().__init__(self.AMOEBA_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class PNAS(NDS):
#     __doc__ = """Search space proposed in
#     `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
#     """ + _INIT_PARAMETER_DOCS
#
#     PNAS_OPS = [
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'conv_7x1_1x7',
#         'skip_connect',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_conv_3x3',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.PNAS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class DARTS(NDS):
#     __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
#     It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
#
#     .. note::
#
#         ``none`` is not included in the operator candidates.
#         It has already been handled in the differentiable implementation of cell.
#
#     """ + _INIT_PARAMETER_DOCS
#
#     DARTS_OPS = [
#         # 'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.DARTS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=4,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#     @classmethod
#     def load_searched_model(
#         cls, name: str,
#         pretrained: bool = False, download: bool = False, progress: bool = True
#     ) -> nn.Module:
#
#         init_kwargs = {}  # all default
#
#         if name == 'darts-v2':
#             init_kwargs.update(
#                 num_cells=20,
#                 width=36,
#             )
#             arch = {
#                 'normal/op_2_0': 'sep_conv_3x3',
#                 'normal/op_2_1': 'sep_conv_3x3',
#                 'normal/input_2_0': 0,
#                 'normal/input_2_1': 1,
#                 'normal/op_3_0': 'sep_conv_3x3',
#                 'normal/op_3_1': 'sep_conv_3x3',
#                 'normal/input_3_0': 0,
#                 'normal/input_3_1': 1,
#                 'normal/op_4_0': 'sep_conv_3x3',
#                 'normal/op_4_1': 'skip_connect',
#                 'normal/input_4_0': 1,
#                 'normal/input_4_1': 0,
#                 'normal/op_5_0': 'skip_connect',
#                 'normal/op_5_1': 'dil_conv_3x3',
#                 'normal/input_5_0': 0,
#                 'normal/input_5_1': 2,
#                 'reduce/op_2_0': 'max_pool_3x3',
#                 'reduce/op_2_1': 'max_pool_3x3',
#                 'reduce/input_2_0': 0,
#                 'reduce/input_2_1': 1,
#                 'reduce/op_3_0': 'skip_connect',
#                 'reduce/op_3_1': 'max_pool_3x3',
#                 'reduce/input_3_0': 2,
#                 'reduce/input_3_1': 1,
#                 'reduce/op_4_0': 'max_pool_3x3',
#                 'reduce/op_4_1': 'skip_connect',
#                 'reduce/input_4_0': 0,
#                 'reduce/input_4_1': 2,
#                 'reduce/op_5_0': 'skip_connect',
#                 'reduce/op_5_1': 'max_pool_3x3',
#                 'reduce/input_5_0': 2,
#                 'reduce/input_5_1': 1
#             }
#
#         else:
#             raise ValueError(f'Unsupported architecture with name: {name}')
#
#         model_factory = cls.fixed_arch(arch)
#         model = model_factory(**init_kwargs)
#
#         if pretrained:
#             weight_file = load_pretrained_weight(name, download=download, progress=progress)
#             pretrained_weights = torch.load(weight_file)
#             model.load_state_dict(pretrained_weights)
#
#         return model
# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.
#
# """File containing NASNet-series search space.
#
# The implementation is based on NDS.
# It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
# """
#
# from functools import partial
# from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast
# import torch.nn.functional as F
#
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal
#
# import torch
#
# import nni.nas.nn.pytorch as nn
# from nni.nas import model_wrapper
#
# from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
# from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
#
# from .utils.fixed import FixedFactory
# from .utils.pretrained import load_pretrained_weight
#
#
# # the following are NAS operations from
# # https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py
#
# OPS = {
#     'none': lambda C, stride, affine:Zero(stride),
#     'avg_pool_2x2': lambda C, stride, affine:nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
#     'avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'avg_pool_5x5': lambda C, stride, affine:nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
#     'max_pool_2x2': lambda C, stride, affine:nn.MaxPool2d(2, stride=stride, padding=0),
#     'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(3, stride=stride, padding=1),
#     'max_pool_5x5': lambda C, stride, affine:nn.MaxPool2d(5, stride=stride, padding=2),
#     'max_pool_7x7': lambda C, stride, affine:nn.MaxPool2d(7, stride=stride, padding=3),
#     'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C),
#     # 'skip_connect': lambda C, stride, affine:nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C, affine=affine),
#     'conv_1x1': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(C, affine=affine)),
#
#     'sep_conv_3x3': lambda C, stride, affine:PrivSepConv(C, C, 3, stride, 1, relu()),
#     'sep_conv_5x5': lambda C, stride, affine:PrivSepConv(C, C, 5, stride, 2, relu()),
#
#     # 'sep_conv_7x7': lambda C, stride, affine:SepConv(C, C, 7, stride, 3, affine=affine),
#     'sep_conv_7x7': lambda C, stride, affine: PrivSepConv(C, C, 7, stride, 3, relu()),
#     'dil_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
#     'dil_conv_5x5': lambda C, stride, affine:PrivDilConv(C, C, 5, stride, 4, 2, relu()),
#
#     'dil_sep_conv_3x3': lambda C, stride, affine:DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
#     # 'conv_3x1_1x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     # 'conv_7x1_1x7': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
#     'conv_3x1_1x3': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
#                                                             nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1),
#                                                                       bias=False),
#                                                             nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0),
#                                                                       bias=False), GN(C)),
#     'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
#                                                             nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3),
#                                                                       bias=False),
#                                                             nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0),
#                                                                       bias=False), GN(C)),
# #############################################################################################################################################################################################
#     'priv_skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'priv_avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#     'priv_max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#     'priv_sep_conv_3x3_relu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, relu()),
#     'priv_sep_conv_3x3_elu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, elu()),
#     'priv_sep_conv_3x3_tanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, tanh()),
#     'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, sigmoid()),
#     'priv_sep_conv_3x3_selu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, selu()),
#     'priv_sep_conv_3x3_htanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, htanh()),
#     'priv_sep_conv_3x3_linear': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, Identity()),
# }
#
# class PrivSepConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super(PrivSepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
#                       bias=False, groups=C_out),
#             GN(C_out),
#             Act,
#         )
#
#     def forward(self, x):
#         x = self.op(x)
#         return x
#
# class StdSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
#         super().__init__(
#             Act,
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             GN(C_in),
#             Act,
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out),
#         )
#
# class PrivFactorizedReduce(nn.Module):
#     def __init__(self, C_in, C_out, Act=None):
#         super(PrivFactorizedReduce, self).__init__()
#         assert C_out % 2 == 0
#         self.relu = Act
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
#                                 padding=0, bias=False)
#         self.bn = GN(C_out)
#
#     def forward(self, x):
#         if self.relu is not None:
#             x = self.relu(x)
#         if x.size(2)%2!=0:
#             x = F.pad(x, (1,0,1,0), "constant", 0)
#
#         out1 = self.conv_1(x)
#         out2 = self.conv_2(x[:, :, 1:, 1:])
#
#         out = torch.cat([out1, out2], dim=1)
#         out = self.bn(out)
#         return out
#
# class PrivDilConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
#         super(PrivDilConv, self).__init__()
#         self.op = nn.Sequential(
#             Act,
#             nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
#                       padding=padding, dilation=dilation, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             GN(C_out)
#         )
#
#     def forward(self, x):
#         return self.op(x)
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
# def GN(plane):
#     return nn.GroupNorm(8, plane, affine=False)
#
# def relu():
#     return nn.ReLU()
#
# def elu():
#     return nn.ELU()
#
# def tanh():
#     return nn.Tanh()
#
# def htanh():
#     return nn.Hardtanh()
#
# def sigmoid():
#     return nn.Sigmoid()
#
# def selu():
#     return nn.SELU()
#
# class ReLUConvBN(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_out, kernel_size, stride=stride,
#                 padding=padding, bias=False
#             ),
#             # nn.BatchNorm2d(C_out, affine=affine)
#             GN(C_out)
#         )
#
#
# class DilConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class SepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#
# class DilSepConv(nn.Sequential):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=stride,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             #nn.BatchNorm2d(C_in, affine=affine),
#             GN(C_in),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(
#                 C_in, C_in, kernel_size=kernel_size, stride=1,
#                 padding=padding, dilation=dilation, groups=C_in, bias=False
#             ),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             #nn.BatchNorm2d(C_out, affine=affine),
#             GN(C_out),
#         )
#
#
# class Zero(nn.Module):
#
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)
#
#
# class FactorizedReduce(nn.Module):
#
#     def __init__(self, C_in, C_out, affine=True):
#         super().__init__()
#         if isinstance(C_out, int):
#             assert C_out % 2 == 0
#         else:   # is a value choice
#             assert all(c % 2 == 0 for c in C_out.all_options())
#         self.relu = nn.ReLU(inplace=False)
#         self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         # self.bn = nn.BatchNorm2d(C_out, affine=affine)
#         self.bn = GN(C_out)
#         self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
#
#     def forward(self, x):
#         x = self.relu(x)
#         y = self.pad(x)
#         out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
#         out = self.bn(out)
#         return out
#
#
# class DropPath_(nn.Module):
#     # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
#     def __init__(self, drop_prob=0.):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.training and self.drop_prob > 0.:
#             keep_prob = 1. - self.drop_prob
#             mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
#             return x.div(keep_prob).mul(mask)
#         return x
#
#
# class AuxiliaryHead(nn.Module):
#     def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
#         super().__init__()
#         if dataset == 'imagenet':
#             # assuming input size 14x14
#             stride = 2
#         elif dataset == 'cifar':
#             stride = 3
#
#         self.features = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
#             nn.Conv2d(C, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 768, 2, bias=False),
#             nn.BatchNorm2d(768),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(768, num_labels)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x
#
#
# class CellPreprocessor(nn.Module):
#     """
#     Aligning the shape of predecessors.
#     是一个用于细胞（cell）构建过程中对输入进行预处理的类。它主要用于对细胞的前驱节点进行形状对齐操作
#     If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
#     See :class:`CellBuilder` on how to calculate those channel numbers.
#     """
#
#     def __init__(self, C_pprev: nn.MaybeChoice[int], C_prev: nn.MaybeChoice[int], C: nn.MaybeChoice[int], last_cell_reduce: bool) -> None:
#         super().__init__()
#
#         if last_cell_reduce:
#             self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
#         else:
#             self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
#         self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)
#
#     def forward(self, cells):
#         assert len(cells) == 2
#         pprev, prev = cells
#         pprev = self.pre0(pprev)
#         prev = self.pre1(prev)
#
#         return [pprev, prev]
#
#
# class CellPostprocessor(nn.Module):
#     """
#     The cell outputs previous cell + this cell, so that cells can be directly chained.
#     """
#
#     def forward(self, this_cell, previous_cells):
#         return [previous_cells[-1], this_cell]
#
#
# class CellBuilder:
#     """The cell builder is used in Repeat.
#     Builds an cell each time it's "called".
#     Note that the builder is ephemeral, it can only be called once for every index.
#     """
#
#     def __init__(self, op_candidates: List[str],
#                  C_prev_in: nn.MaybeChoice[int],
#                  C_in: nn.MaybeChoice[int],
#                  C: nn.MaybeChoice[int],
#                  num_nodes: int,
#                  merge_op: Literal['all', 'loose_end'],
#                  first_cell_reduce: bool, last_cell_reduce: bool,
#                  drop_path_prob: float):
#         self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
#         self.C_in = C_in                # This is the out channesl of last cell.
#         self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
#         self.op_candidates = op_candidates
#         self.num_nodes = num_nodes
#         self.merge_op: Literal['all', 'loose_end'] = merge_op
#         self.first_cell_reduce = first_cell_reduce
#         self.last_cell_reduce = last_cell_reduce
#         self.drop_path_prob = drop_path_prob
#         self._expect_idx = 0
#
#         # It takes an index that is the index in the repeat.
#         # Number of predecessors for each cell is fixed to 2.
#         self.num_predecessors = 2
#
#         # Number of ops per node is fixed to 2.
#         self.num_ops_per_node = 2
# #这个方法负责生成操作（operation）。在这个方法中，首先根据cell的类型（即是否为降采样cell）和输入节点的索引确定stride，然后根据指定的操作名和通道数创建operation。
#     def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,op: str, channels: int, is_reduction_cell: bool):
#         if is_reduction_cell and (input_index is None or input_index < self.num_predecessors):  # could be none when constructing search space
#             stride = 2
#         else:
#             stride = 1
#         operation = OPS[op](channels, stride, True)
#         if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
#             # Omit drop-path when operation is skip connect.
#             # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
#             return nn.Sequential(operation, DropPath_(self.drop_path_prob))
#         return operation
#
#     #CellBuilder中的__call__方法用于实际构建cell。在该方法中，根据当前的repeat_idx确定是否为降采样的cell，并创建CellPreprocessor对象。
#     def __call__(self, repeat_idx: int):#在构建多个细胞的过程中，可以使用 repeat_idx 来跟踪当前正在构建的细胞是第几个重复的细胞
#         if self._expect_idx != repeat_idx:
#             raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')
#
#         # Reduction cell means stride = 2 and channel multiplied by 2.判断是否为降采样cell。
#         is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce
#
#         # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
#         #创建一个CellPreprocessor对象，用于处理cell的输入和输出通道。
#         preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)
#         #用于存储操作的工厂函数。工厂函数根据指定的操作名称和其他参数来创建操作对象，其中包括使用self.op_factory方法创建的操作对象。
#         #创建一个字典ops_factory，存储所有可能的操作和对应的工厂函数。工厂函数根据操作名和其他参数创建operation。
#         ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
#         for op in self.op_candidates:
#             ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)
#         #对象包含了一组操作节点，形成了一个完整的Cell。
#         cell = nn.Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
#                        preprocessor=preprocessor, postprocessor=CellPostprocessor(),
#                        label='reduce' if is_reduction_cell else 'normal')
#
#         # update state
#         self.C_prev_in = self.C_in
#         self.C_in = self.C * len(cell.output_node_indices)
#         self.last_cell_reduce = is_reduction_cell
#         self._expect_idx += 1
#         return cell
#
# #NDSStage包含多个Cell，通过继承nn.Repeat类来实现。nn.Repeat允许多次重复执行同一个操作，因此可以方便地构建多个Cell。
# class NDSStage(nn.Repeat):
#     """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.
#
#     In NDS, we can't simply use Repeat to stack the blocks,
#     because the output shape of each stacked block can be different.
#     This is a problem for one-shot strategy because they assume every possible candidate
#     should return values of the same shape.
#
#     Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
#     to manually align the shapes -- specifically, to transform the first block in each stage.
#
#     This is not required though, when depth is not changing, or the mutable depth causes no problem
#     (e.g., when the minimum depth is large enough).
#
#     .. attention::
#
#        Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
#        which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
#     """
#
#     estimated_out_channels_prev: int
#     """Output channels of cells in last stage.表示上一个阶段中的Cell的输出通道数"""
#
#     estimated_out_channels: int
#     """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``.表示该阶段中的Cell的输出通道数"""
#
#     downsampling: bool
#     """This stage has downsampling表示该阶段是否进行降采样"""
#
#     #用于创建一个变换模块，用于将第一个Cell的输出形状与该阶段中其他Cell的输出形状对齐。
#     def first_cell_transformation_factory(self) -> Optional[nn.Module]:
#         """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
#         if self.downsampling:
#             return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
#         elif self.estimated_out_channels_prev is not self.estimated_out_channels:
#             # Can't use != here, ValueChoice doesn't support
#             return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
#         return None
#
#
# class NDSStagePathSampling(PathSamplingRepeat):
#     """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in sampled or self.first_cell_transformation is None:
#             return super().reduction(items, sampled)
#         # items[0] must be the result of first cell
#         assert len(items[0]) == 2
#         # Only apply the transformation on "prev" output.
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, sampled)
#
#
# class NDSStageDifferentiable(DifferentiableMixedRepeat):
#     """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
#     @classmethod
#     def mutate(cls, module, name, memo, mutate_kwargs):
#         if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
#             # Only interesting when depth is mutable
#             softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
#             return cls(
#                 module.first_cell_transformation_factory(),
#                 cast(List[nn.Module], module.blocks),
#                 module.depth_choice,
#                 softmax,
#                 memo
#             )
#
#     def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.first_cell_transformation = first_cell_transformation
#
#     def reduction(
#         self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if 1 not in depths or self.first_cell_transformation is None:
#             return super().reduction(items, weights, depths)
#         # Same as NDSStagePathSampling
#         assert len(items[0]) == 2
#         items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
#         return super().reduction(items, weights, depths)
#
#
# _INIT_PARAMETER_DOCS = """
#
#     Notes
#     -----
#
#     To use NDS spaces with one-shot strategies,
#     especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
#     please use :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
#     and :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
#     This is because the output shape of each stacked block in :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStage` can be different.
#     For example::
#
#         from nni.retiarii.hub.pytorch.nasnet import NDSStageDifferentiable
#         darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])
#
#     Parameters
#     ----------
#     width
#         A fixed initial width or a tuple of widths to choose from.
#     num_cells
#         A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
#     dataset
#         The essential differences are in "stem" cells, i.e., how they process the raw image input.
#         Choosing "imagenet" means more downsampling at the beginning of the network.
#     auxiliary_loss
#         If true, another auxiliary classification head will produce the another prediction.
#         This makes the output of network two logits in the training phase.
#     drop_path_prob
#         Apply drop path. Enabled when it's set to be greater than 0.
#
# """
#
#
# class NDS(nn.Module):
#     __doc__ = """
#     The unified version of NASNet search space.
#
#     We follow the implementation in
#     `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
#     See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.
#
#     Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
#     ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.
#
#     NDS has a speciality that it has mutable depths/widths.
#     This is implemented by accepting a list of int as ``num_cells`` / ``width``.
#     """ + _INIT_PARAMETER_DOCS.rstrip() + """
#     op_candidates
#         List of operator candidates. Must be from ``OPS``.
#     merge_op
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     num_nodes_per_cell
#         See :class:`~nni.retiarii.nn.pytorch.Cell`.
#     """
#
#     def __init__(self,
#                  op_candidates: List[str],
#                  merge_op: Literal['all', 'loose_end'] = 'all',
#                  num_nodes_per_cell: int = 4,
#                  width: Union[Tuple[int, ...], int] = 16,
#                  num_cells: Union[Tuple[int, ...], int] = 20,#6
#                  dataset: Literal['cifar', 'imagenet'] = 'imagenet',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__()
#
#         self.dataset = dataset
#         self.num_labels = 10 if dataset == 'cifar' else 1000
#         self.auxiliary_loss = auxiliary_loss
#         self.drop_path_prob = drop_path_prob
# #如果num_cells是一个列表，那么使用ValueChoice来表示这是一个候选值的集合，后续可以从中选择一个值。否则，直接使用num_cells作为网络的深度。
#         # preprocess the specified width and depth
#         if isinstance(width, Iterable):
#             C = nn.ValueChoice(list(width), label='width')
#         else:
#             C = width
#
#         self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
#         if isinstance(num_cells, Iterable):
#             self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
#         #[2,2,2]每个阶段（stage）中重复构建的Cell的数量
#         num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]
#
#         # auxiliary head is different for network targetted at different datasets
#         if dataset == 'imagenet':
#             self.stem0 = nn.Sequential(
#                 nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(cast(int, C // 2)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             self.stem1 = nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(C),
#             )
#             C_pprev = C_prev = C_curr = C
#             last_cell_reduce = True
#         elif dataset == 'cifar':
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
#                 #nn.BatchNorm2d(cast(int, 3 * C))
#                 GN(cast(int, 3 * C))
#             )
#             C_pprev = C_prev = 3 * C
#             C_curr = C
#             last_cell_reduce = False
#         else:
#             raise ValueError(f'Unsupported dataset: {dataset}')
#
#         self.stages = nn.ModuleList()
#         #stage_idx > 0 作为是否对输入执行降采样（downsampling）的判断条件。stage_idx 为 0 代表网络的第一阶段，在这个阶段不执行降采样。而 stage_idx > 0 的情况（即第二阶段和第三阶段），在创建cell时会进行降采样。
#         for stage_idx in range(3):
#             if stage_idx > 0:
#                 C_curr *= 2
#             #NDSStage的构造函数接受一个CellBuilder对象和一个整数num_cells作为参数。CellBuilder用于构建单个Cell，num_cells表示在该阶段中要构建的Cell的数量。
#             cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
#             #在循环中，创建了三个阶段的cells。每个阶段的通道数C_curr可能会翻倍，对于第一个阶段，会创建一个CellBuilder对象，然后用这个CellBuilder对象创建一个NDSStage对象。这个NDSStage对象代表了一个阶段，其中包含了多个cell。它也记录了输入和输出的通道数，以及是否进行下采样。
#             stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])
#
#             if isinstance(stage, NDSStage):
#                 stage.estimated_out_channels_prev = cast(int, C_prev)
#                 stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
#                 stage.downsampling = stage_idx > 0
#
#             self.stages.append(stage)
# #在每个阶段的结束，更新了输入通道数（C_pprev和C_prev）以供下一个阶段使用，并设置了最后一个cell是否是下采样cell。
#             # NOTE: output_node_indices will be computed on-the-fly in trial code.
#             # When constructing model space, it's just all the nodes in the cell,
#             # which happens to be the case of one-shot supernet.
#
#             # C_pprev is output channel number of last second cell among all the cells already built.
#             if len(stage) > 1:
#                 # Contains more than one cell
#                 C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
#             else:
#                 # Look up in the out channels of last stage.
#                 C_pprev = C_prev
#
#             # This was originally,
#             # C_prev = num_nodes_per_cell * C_curr.
#             # but due to loose end, it becomes,
#             C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr
#
#             # Useful in aligning the pprev and prev cell.
#             last_cell_reduce = cell_builder.last_cell_reduce
#
#             if stage_idx == 2:
#                 C_to_auxiliary = C_prev
# #如果辅助损失被启用，那么在最后一个阶段创建了一个AuxiliaryHead。
#         if auxiliary_loss:
#             assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
#             self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore
#
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
#
#     def forward(self, inputs):
#         if self.dataset == 'imagenet':
#             s0 = self.stem0(inputs)
#             s1 = self.stem1(s0)
#         else:
#             s0 = s1 = self.stem(inputs)
# #这段代码实现了在网络的第三个阶段使用辅助损失来提高网络的训练效果。辅助损失是一种常用的技术，用于在网络的深层添加额外的损失函数，从而减少梯度消失问题，提高网络的性能。
#         for stage_idx, stage in enumerate(self.stages):
#             if stage_idx == 2 and self.auxiliary_loss and self.training:
#                 assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
#                 for block_idx, block in enumerate(stage):
#                     # auxiliary loss is attached to the first cell of the last stage.
#                     s0, s1 = block([s0, s1])
#                     if block_idx == 0:
#                         logits_aux = self.auxiliary_head(s1)
#             else:
#                 s0, s1 = stage([s0, s1])
#
#         out = self.global_pooling(s1)
#
#         logits = self.classifier(out.view(out.size(0), -1))
#         if self.training and self.auxiliary_loss:
#             return logits, logits_aux  # type: ignore
#         else:
#             return logits
#
#     def set_drop_path_prob(self, drop_prob):
#         """
#         Set the drop probability of Drop-path in the network.
#         Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
#         """
#         for module in self.modules():
#             if isinstance(module, DropPath_):
#                 module.drop_prob = drop_prob
#
#     @classmethod
#     def fixed_arch(cls, arch: dict) -> FixedFactory:
#         return FixedFactory(cls, arch)
#
#
# @model_wrapper
# class NASNet(NDS):
#     __doc__ = """
#     Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~NASNet.NASNET_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     NASNET_OPS = [
#         'skip_connect',
#         'conv_3x1_1x3',
#         'conv_7x1_1x7',
#         'dil_conv_3x3',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'max_pool_5x5',
#         'max_pool_7x7',
#         'conv_1x1',
#         'conv_3x3',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.NASNET_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class ENAS(NDS):
#     __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~ENAS.ENAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     # ENAS_OPS = [
#     #     'skip_connect',
#     #     'sep_conv_3x3',
#     #     'sep_conv_5x5',
#     #     'avg_pool_3x3',
#     #     'max_pool_3x3',
#     # ]
#     ENAS_OPS = [
#         'none',
#         'priv_max_pool_3x3',
#         'priv_avg_pool_3x3',
#         'priv_skip_connect',
#         'priv_sep_conv_3x3_relu',
#         'priv_sep_conv_3x3_selu',
#         'priv_sep_conv_3x3_tanh',
#         'priv_sep_conv_3x3_linear',
#         'priv_sep_conv_3x3_htanh',
#         'priv_sep_conv_3x3_sigmoid',
#     ]
#     """The candidate operations."""
# #我们希望在子类中保留父类方法的行为，可以使用super()函数来调用父类的方法并在其基础上进行修改
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         #在初始化过程中，它调用了父类NDS的__init__方法，并传递了一些参数，用于构建基本的模型架构搜索空间。
#         super().__init__(self.ENAS_OPS,
#                          merge_op='all',#原本是loose_end
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class AmoebaNet(NDS):
#     __doc__ = """Search space proposed in
#     `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~AmoebaNet.AMOEBA_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
#     """ + _INIT_PARAMETER_DOCS
#
#     AMOEBA_OPS = [
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_sep_conv_3x3',
#         'conv_7x1_1x7',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#
#         super().__init__(self.AMOEBA_OPS,
#                          merge_op='loose_end',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class PNAS(NDS):
#     __doc__ = """Search space proposed in
#     `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
#     It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
#     """ + _INIT_PARAMETER_DOCS
#
#     PNAS_OPS = [
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'sep_conv_7x7',
#         'conv_7x1_1x7',
#         'skip_connect',
#         'avg_pool_3x3',
#         'max_pool_3x3',
#         'dil_conv_3x3',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.PNAS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=5,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#
# @model_wrapper
# class DARTS(NDS):
#     __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.
#
#     It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
#     Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
#     It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
#
#     .. note::
#
#         ``none`` is not included in the operator candidates.
#         It has already been handled in the differentiable implementation of cell.
#
#     """ + _INIT_PARAMETER_DOCS
#
#     DARTS_OPS = [
#         # 'none',
#         'max_pool_3x3',
#         'avg_pool_3x3',
#         'skip_connect',
#         'sep_conv_3x3',
#         'sep_conv_5x5',
#         'dil_conv_3x3',
#         'dil_conv_5x5',
#     ]
#     """The candidate operations."""
#
#     def __init__(self,
#                  width: Union[Tuple[int, ...], int] = (16, 24, 32),
#                  num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
#                  dataset: Literal['cifar', 'imagenet'] = 'cifar',
#                  auxiliary_loss: bool = False,
#                  drop_path_prob: float = 0.):
#         super().__init__(self.DARTS_OPS,
#                          merge_op='all',
#                          num_nodes_per_cell=4,
#                          width=width,
#                          num_cells=num_cells,
#                          dataset=dataset,
#                          auxiliary_loss=auxiliary_loss,
#                          drop_path_prob=drop_path_prob)
#
#     @classmethod
#     def load_searched_model(
#         cls, name: str,
#         pretrained: bool = False, download: bool = False, progress: bool = True
#     ) -> nn.Module:
#
#         init_kwargs = {}  # all default
#
#         if name == 'darts-v2':
#             init_kwargs.update(
#                 num_cells=20,
#                 width=36,
#             )
#             arch = {
#                 'normal/op_2_0': 'sep_conv_3x3',
#                 'normal/op_2_1': 'sep_conv_3x3',
#                 'normal/input_2_0': 0,
#                 'normal/input_2_1': 1,
#                 'normal/op_3_0': 'sep_conv_3x3',
#                 'normal/op_3_1': 'sep_conv_3x3',
#                 'normal/input_3_0': 0,
#                 'normal/input_3_1': 1,
#                 'normal/op_4_0': 'sep_conv_3x3',
#                 'normal/op_4_1': 'skip_connect',
#                 'normal/input_4_0': 1,
#                 'normal/input_4_1': 0,
#                 'normal/op_5_0': 'skip_connect',
#                 'normal/op_5_1': 'dil_conv_3x3',
#                 'normal/input_5_0': 0,
#                 'normal/input_5_1': 2,
#                 'reduce/op_2_0': 'max_pool_3x3',
#                 'reduce/op_2_1': 'max_pool_3x3',
#                 'reduce/input_2_0': 0,
#                 'reduce/input_2_1': 1,
#                 'reduce/op_3_0': 'skip_connect',
#                 'reduce/op_3_1': 'max_pool_3x3',
#                 'reduce/input_3_0': 2,
#                 'reduce/input_3_1': 1,
#                 'reduce/op_4_0': 'max_pool_3x3',
#                 'reduce/op_4_1': 'skip_connect',
#                 'reduce/input_4_0': 0,
#                 'reduce/input_4_1': 2,
#                 'reduce/op_5_0': 'skip_connect',
#                 'reduce/op_5_1': 'max_pool_3x3',
#                 'reduce/input_5_0': 2,
#                 'reduce/input_5_1': 1
#             }
#
#         else:
#             raise ValueError(f'Unsupported architecture with name: {name}')
#
#         model_factory = cls.fixed_arch(arch)
#         model = model_factory(**init_kwargs)
#
#         if pretrained:
#             weight_file = load_pretrained_weight(name, download=download, progress=progress)
#             pretrained_weights = torch.load(weight_file)
#             model.load_state_dict(pretrained_weights)
#
#         return model


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



from functools import partial
from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast
import torch.nn.functional as F

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch

import nni.nas.nn.pytorch as nn
from nni.nas import model_wrapper

from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat

from .utils.fixed import FixedFactory
from .utils.pretrained import load_pretrained_weight


# the following are NAS operations from
# https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py
OPS = {
    'none': lambda C, stride, affine:Zero(stride),
    'avg_pool_2x2': lambda C, stride, affine:nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
    'avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5x5': lambda C, stride, affine:nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
    'max_pool_2x2': lambda C, stride, affine:nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda C, stride, affine:nn.MaxPool2d(5, stride=stride, padding=2),
    'max_pool_7x7': lambda C, stride, affine:nn.MaxPool2d(7, stride=stride, padding=3),
    'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C),
    # 'skip_connect': lambda C, stride, affine:nn.Identity() if stride == 1 else PrivFactorizedReduce(C, C, affine=affine),
    'conv_1x1': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),nn.BatchNorm2d(C, affine=affine)),
    'conv_3x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(C, affine=affine)),

    'sep_conv_3x3': lambda C, stride, affine:StdSepConv(C, C, 3, stride, 1, relu()),
    'sep_conv_5x5': lambda C, stride, affine:StdSepConv(C, C, 5, stride, 2, relu()),

    # 'sep_conv_7x7': lambda C, stride, affine:SepConv(C, C, 7, stride, 3, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: StdSepConv(C, C, 7, stride, 3, relu()),
    'dil_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
    'dil_conv_5x5': lambda C, stride, affine:PrivDilConv(C, C, 5, stride, 4, 2, relu()),

    # 'dil_sep_conv_3x3': lambda C, stride, affine:DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_sep_conv_3x3': lambda C, stride, affine:PrivDilConv(C, C, 3, stride, 2, 2, relu()),
    # 'conv_3x1_1x3': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
    # 'conv_7x1_1x7': lambda C, stride, affine:nn.Sequential(nn.ReLU(inplace=False),nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),nn.BatchNorm2d(C, affine=affine)),
    'conv_3x1_1x3': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
                                                            nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1),
                                                                      bias=False),
                                                            nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0),
                                                                      bias=False), GN(C)),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False),
                                                            nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3),
                                                                      bias=False),
                                                            nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0),
                                                                      bias=False), GN(C)),
#############################################################################################################################################################################################
    'priv_skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'priv_avg_pool_3x3': lambda C, stride, affine:nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'priv_max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'priv_sep_conv_3x3_relu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, relu()),
    'priv_sep_conv_3x3_elu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, elu()),
    'priv_sep_conv_3x3_tanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, tanh()),
    'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, sigmoid()),
    'priv_sep_conv_3x3_selu': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, selu()),
    'priv_sep_conv_3x3_htanh': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, htanh()),
    'priv_sep_conv_3x3_linear': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, Identity()),
    'priv_sep_conv_3x3_softsign': lambda C, stride, affine: PrivSepConv(C, C, 3, stride, 1, softsign()),
}

class PrivSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False, groups=C_out),
            GN(C_out),
            Act,
        )

    def forward(self, x):
        x = self.op(x)
        return x

class StdSepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super().__init__(
            Act,
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            GN(C_in),
            Act,
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            GN(C_out),
        )

class PrivFactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, Act=None):
        super(PrivFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = Act
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = GN(C_out)

    def forward(self, x):
        if self.relu is not None:
            x = self.relu(x)
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)

        out1 = self.conv_1(x)
        out2 = self.conv_2(x[:, :, 1:, 1:])

        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out

class PrivDilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
        super(PrivDilConv, self).__init__()
        self.op = nn.Sequential(
            Act,
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            GN(C_out)
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def GN(plane):
    return nn.GroupNorm(8, plane, affine=False)

def relu():
    return nn.ReLU()

def elu():
    return nn.ELU()

def tanh():
    return nn.Tanh()

def htanh():
    return nn.Hardtanh()

def softsign():
    return nn.Softsign()

def sigmoid():
    return nn.Sigmoid()

def selu():
    return nn.SELU()

class ReLUConvBN(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),#本来是有的
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride,
                padding=padding, bias=False
            ),
            # nn.BatchNorm2d(C_out, affine=affine)
            GN(C_out)
        )


class DilConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out, affine=affine),
            GN(C_out)
        )


class SepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_in, affine=affine),
            GN(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out, affine=affine),
            GN(C_out),
        )


class DilSepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_in, affine=affine),
            GN(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out, affine=affine),
            GN(C_out),
        )


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        if isinstance(C_out, int):
            assert C_out % 2 == 0
        else:   # is a value choice
            assert all(c % 2 == 0 for c in C_out.all_options())
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.bn = GN(C_out)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.zeros((x.size(0), 1, 1, 1), dtype=torch.float, device=x.device).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
        super().__init__()
        if dataset == 'imagenet':
            # assuming input size 14x14
            stride = 2
        elif dataset == 'cifar':
            stride = 3

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            # nn.BatchNorm2d(128),
            GN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # nn.BatchNorm2d(768),
            GN(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class CellPreprocessor(nn.Module):
    """
    Aligning the shape of predecessors.
    是一个用于细胞（cell）构建过程中对输入进行预处理的类。它主要用于对细胞的前驱节点进行形状对齐操作
    If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
    See :class:`CellBuilder` on how to calculate those channel numbers.
    """

    def __init__(self, C_pprev: nn.MaybeChoice[int], C_prev: nn.MaybeChoice[int], C: nn.MaybeChoice[int], last_cell_reduce: bool) -> None:
        super().__init__()

        if last_cell_reduce:
            self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
        else:
            self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
        self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)

    def forward(self, cells):
        assert len(cells) == 2
        pprev, prev = cells
        pprev = self.pre0(pprev)
        prev = self.pre1(prev)

        return [pprev, prev]


class CellPostprocessor(nn.Module):
    """
    The cell outputs previous cell + this cell, so that cells can be directly chained.
    """

    def forward(self, this_cell, previous_cells):
        return [previous_cells[-1], this_cell]

class MaxPool2x2(nn.Module):
    def __init__(self, reduction=True):
        super(MaxPool2x2, self).__init__()
        self.last_cell_reduce = reduction

        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

    def forward(self, s1):
        return self.layers(s1)


# class MaxPoolCell(nn.Module):
#     def __init__(self, kernel_size=2, stride=2):
#         super(MaxPoolCell, self).__init__()
#         self.maxpool = nn.MaxPool2d(kernel_size, stride)
#
#     def forward(self, x):
#         return self.maxpool(x)

class CellBuilder:
    """The cell builder is used in Repeat.
    Builds an cell each time it's "called".
    Note that the builder is ephemeral, it can only be called once for every index.
    """

    def __init__(self, op_candidates: List[str],
                 C_prev_in: nn.MaybeChoice[int],
                 C_in: nn.MaybeChoice[int],
                 C: nn.MaybeChoice[int],
                 num_nodes: int,
                 merge_op: Literal['all', 'loose_end'],
                 first_cell_reduce: bool, last_cell_reduce: bool,
                 drop_path_prob: float):
        self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
        self.C_in = C_in                # This is the out channesl of last cell.
        self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
        self.op_candidates = op_candidates
        self.num_nodes = num_nodes
        self.merge_op: Literal['all', 'loose_end'] = merge_op
        self.first_cell_reduce = first_cell_reduce
        self.last_cell_reduce = last_cell_reduce
        self.drop_path_prob = drop_path_prob
        self._expect_idx = 0

        # It takes an index that is the index in the repeat.
        # Number of predecessors for each cell is fixed to 2.
        self.num_predecessors = 2

        # Number of ops per node is fixed to 2.
        self.num_ops_per_node = 2

#这个方法负责生成操作（operation）。在这个方法中，首先根据cell的类型（即是否为降采样cell）和输入节点的索引确定stride，然后根据指定的操作名和通道数创建operation。
    def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,op: str, channels: int, is_reduction_cell: bool):
        if is_reduction_cell and (input_index is None or input_index < self.num_predecessors):  # could be none when constructing search space
            stride = 2
            #####new add 这样只会让降采样单元都会变成pooling########
            #operation = OPS['priv_max_pool_3x3'](channels, stride, True)
            #############
        else:
            stride = 1
        operation = OPS[op](channels, stride, True)
        if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
            # Omit drop-path when operation is skip connect.
            # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
            return nn.Sequential(operation, DropPath_(self.drop_path_prob))
        return operation

    #CellBuilder中的__call__方法用于实际构建cell。在该方法中，根据当前的repeat_idx确定是否为降采样的cell，并创建CellPreprocessor对象。
    def __call__(self, repeat_idx: int):#在构建多个细胞的过程中，可以使用 repeat_idx 来跟踪当前正在构建的细胞是第几个重复的细胞
        # if self._expect_idx != repeat_idx or repeat_idx == 1:
        #     #raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')
        #     return None
        if self._expect_idx != repeat_idx :
            raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')

        # Reduction cell means stride = 2 and channel multiplied by 2.判断是否为降采样cell。
        is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce

        if  self.last_cell_reduce :
            self.C_prev_in = self.C_in
            self.C_in = self.C #* 2  # update the number of output channels as it's a reduction cell
            self.last_cell_reduce = is_reduction_cell
            self._expect_idx += 1

            cell = MaxPool2x2()
            return cell
        else:
            # if is_reduction_cell:
            #     self._expect_idx += 1
            #     return nn.MaxPool2d(kernel_size=2, stride=2)
            # else:
            # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
            #创建一个CellPreprocessor对象，用于处理cell的输入和输出通道。
            preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)
            #用于存储操作的工厂函数。工厂函数根据指定的操作名称和其他参数来创建操作对象，其中包括使用self.op_factory方法创建的操作对象。
            #创建一个字典ops_factory，存储所有可能的操作和对应的工厂函数。工厂函数根据操作名和其他参数创建operation。
            ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
            for op in self.op_candidates:
                    ######添加的这样只会让降采样单元都会变成pooling#######
                    # if is_reduction_cell:
                    #     if op != 'priv_max_pool_3x3':
                    #         continue
                    #############
                ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)
            #对象包含了一组操作节点，形成了一个完整的Cell。
            if is_reduction_cell == False:
                cell = nn.Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
                                    preprocessor=preprocessor, postprocessor=CellPostprocessor(),
                                    label='reduce' if is_reduction_cell else 'normal')
                # update state
                self.C_prev_in = self.C_in
                self.C_in = self.C * len(cell.output_node_indices)
                self.last_cell_reduce = is_reduction_cell
                self._expect_idx += 1

                return cell

#NDSStage包含多个Cell，通过继承nn.Repeat类来实现。nn.Repeat允许多次重复执行同一个操作，因此可以方便地构建多个Cell。
class NDSStage(nn.Repeat):
    """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.

    In NDS, we can't simply use Repeat to stack the blocks,
    because the output shape of each stacked block can be different.
    This is a problem for one-shot strategy because they assume every possible candidate
    should return values of the same shape.

    Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
    to manually align the shapes -- specifically, to transform the first block in each stage.

    This is not required though, when depth is not changing, or the mutable depth causes no problem
    (e.g., when the minimum depth is large enough).

    .. attention::

       Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
       which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
    """

    estimated_out_channels_prev: int
    """Output channels of cells in last stage.表示上一个阶段中的Cell的输出通道数"""

    estimated_out_channels: int
    """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``.表示该阶段中的Cell的输出通道数"""

    downsampling: bool
    """This stage has downsampling表示该阶段是否进行降采样"""

    #用于创建一个变换模块，用于将第一个Cell的输出形状与该阶段中其他Cell的输出形状对齐。
    def first_cell_transformation_factory(self) -> Optional[nn.Module]:
        """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
        if self.downsampling:
            return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
        elif self.estimated_out_channels_prev is not self.estimated_out_channels:
            # Can't use != here, ValueChoice doesn't support
            return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
        return None


class NDSStagePathSampling(PathSamplingRepeat):
    """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
            return cls(
                module.first_cell_transformation_factory(),
                cast(List[nn.Module], module.blocks),
                module.depth_choice
            )

    def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_cell_transformation = first_cell_transformation

    def reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if 1 not in sampled or self.first_cell_transformation is None:
            return super().reduction(items, sampled)
        # items[0] must be the result of first cell
        assert len(items[0]) == 2
        # Only apply the transformation on "prev" output.
        items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
        return super().reduction(items, sampled)


class NDSStageDifferentiable(DifferentiableMixedRepeat):
    """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, NDSStage) and isinstance(module.depth_choice, nn.choice.ValueChoiceX):
            # Only interesting when depth is mutable
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(
                module.first_cell_transformation_factory(),
                cast(List[nn.Module], module.blocks),
                module.depth_choice,
                softmax,
                memo
            )

    def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_cell_transformation = first_cell_transformation

    def reduction(
        self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if 1 not in depths or self.first_cell_transformation is None:
            return super().reduction(items, weights, depths)
        # Same as NDSStagePathSampling
        assert len(items[0]) == 2
        items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
        return super().reduction(items, weights, depths)


_INIT_PARAMETER_DOCS = """

    Notes
    -----

    To use NDS spaces with one-shot strategies,
    especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
    please use :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
    and :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
    This is because the output shape of each stacked block in :class:`~nni.retiarii.hub.pytorch.nasnet.NDSStage` can be different.
    For example::

        from nni.retiarii.hub.pytorch.nasnet import NDSStageDifferentiable
        darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])

    Parameters
    ----------
    width
        A fixed initial width or a tuple of widths to choose from.
    num_cells
        A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
    dataset
        The essential differences are in "stem" cells, i.e., how they process the raw image input.
        Choosing "imagenet" means more downsampling at the beginning of the network.
    auxiliary_loss
        If true, another auxiliary classification head will produce the another prediction.
        This makes the output of network two logits in the training phase.
    drop_path_prob
        Apply drop path. Enabled when it's set to be greater than 0.

"""


class NDS(nn.Module):
    __doc__ = """
    The unified version of NASNet search space.

    We follow the implementation in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.

    Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
    ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.

    NDS has a speciality that it has mutable depths/widths.
    This is implemented by accepting a list of int as ``num_cells`` / ``width``.
    """ + _INIT_PARAMETER_DOCS.rstrip() + """
    op_candidates
        List of operator candidates. Must be from ``OPS``.
    merge_op
        See :class:`~nni.retiarii.nn.pytorch.Cell`.
    num_nodes_per_cell
        See :class:`~nni.retiarii.nn.pytorch.Cell`.
    """

    def __init__(self,
                 op_candidates: List[str],
                 merge_op: Literal['all', 'loose_end'] = 'loose_end',
                 num_nodes_per_cell: int = 4,
                 width: Union[Tuple[int, ...], int] = 16,
                 num_cells: Union[Tuple[int, ...], int] = 20,#6
                 dataset: Literal['cifar', 'imagenet'] = 'imagenet',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__()

        self.dataset = dataset
        self.num_labels = 10 if dataset == 'cifar' else 1000
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob
#如果num_cells是一个列表，那么使用ValueChoice来表示这是一个候选值的集合，后续可以从中选择一个值。否则，直接使用num_cells作为网络的深度。
        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nn.ValueChoice(list(width), label='width')
        else:
            C = width

        self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
        #[2,2,2]每个阶段（stage）中重复构建的Cell的数量
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

        # auxiliary head is different for network targetted at different datasets
        if dataset == 'imagenet':
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, C // 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
                #nn.BatchNorm2d(cast(int, 3 * C))
                GN(cast(int, 3 * C))
            )
            C_curr = C_pprev = C_prev = 3 * C
            ####################################原本的直接是下面的#############################
            #C_curr = C
            last_cell_reduce = False
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.stages = nn.ModuleList()
        #stage_idx > 0 作为是否对输入执行降采样（downsampling）的判断条件。stage_idx 为 0 代表网络的第一阶段，在这个阶段不执行降采样。而 stage_idx > 0 的情况（即第二阶段和第三阶段），在创建cell时会进行降采样。
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2 #增加通道数这样的设计是为了逐渐增加模型的容量和复杂性。通过逐渐增加通道数，模型可以逐步学习更复杂的特征，并具备更强的表示能力。这有助于提高模型的性能和表达能力。
            #NDSStage的构造函数接受一个CellBuilder对象和一个整数num_cells作为参数。CellBuilder用于构建单个Cell，num_cells表示在该阶段中要构建的Cell的数量。
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            #在循环中，创建了三个阶段的cells。每个阶段的通道数C_curr可能会翻倍，对于第一个阶段，会创建一个CellBuilder对象，然后用这个CellBuilder对象创建一个NDSStage对象。这个NDSStage对象代表了一个阶段，其中包含了多个cell。它也记录了输入和输出的通道数，以及是否进行下采样。
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])
            #尽管CellBuilder在循环中被实例化了很多次，但这并不意味着CellBuilder的__call__方法也被调用了同样多的次数。实际上，CellBuilder的__call__方法的调用次数是由NDSStage内部实现决定的。
            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
                stage.downsampling = stage_idx > 0

            self.stages.append(stage)
#在每个阶段的结束，更新了输入通道数（C_pprev和C_prev）以供下一个阶段使用，并设置了最后一个cell是否是下采样cell。
            # NOTE: output_node_indices will be computed on-the-fly in trial code.
            # When constructing model space, it's just all the nodes in the cell,
            # which happens to be the case of one-shot supernet.

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
                print("The first normal cell C_pprev", C_pprev)
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev
                print("The first normal cell C_prev/C_pprev", C_prev, C_pprev)

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            if cell_builder.first_cell_reduce:
                #C_pprev = C_prev
                C_prev = C_pprev
                print("The second reduction cell C_prev/C_pprev",C_prev,C_pprev)
            else:
                C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr
                print("The second normal cell C_prev/C_pprev", C_prev, C_pprev)
            #C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr


            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev
#如果辅助损失被启用，那么在最后一个阶段创建了一个AuxiliaryHead。
        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        print("After global_pooling C_prev:", C_prev)
        self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
        # self.global_pooling = nn.Sequential(
        #     nn.Conv2d(cast(int, C_prev), 128, 1, bias=False),
        #     GN(128), )
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(8192, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 10), )


    def forward(self, inputs):
        if self.dataset == 'imagenet':
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)
#这段代码实现了在网络的第三个阶段使用辅助损失来提高网络的训练效果。辅助损失是一种常用的技术，用于在网络的深层添加额外的损失函数，从而减少梯度消失问题，提高网络的性能。
        # for stage_idx, stage in enumerate(self.stages):
        #     if stage_idx == 2 and self.auxiliary_loss and self.training:
        #         assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
        #         for block_idx, block in enumerate(stage):
        #             # auxiliary loss is attached to the first cell of the last stage.
        #             s0, s1 = block([s0, s1])
        #             if block_idx == 0:
        #                 logits_aux = self.auxiliary_head(s1)
        #     else:
        #
        #         s0, s1 = stage([s0, s1])
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if (stage_idx == 1 and block_idx == 1) or (stage_idx == 2 and block_idx == 1):
                    #if isinstance(block, nn.MaxPool2d):
                        # Max pooling needs only one input
                        s1 = block(s1)
                        s0 = s1
                else:
                    s0, s1 = block([s0, s1])

        out = self.global_pooling(s1)

        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.auxiliary_loss:
            return logits, logits_aux  # type: ignore
        else:
            return logits

    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob

    @classmethod
    def fixed_arch(cls, arch: dict) -> FixedFactory:
        return FixedFactory(cls, arch)


@model_wrapper
class NASNet(NDS):
    __doc__ = """
    Search space proposed in `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~NASNet.NASNET_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    NASNET_OPS = [
        'skip_connect',
        'conv_3x1_1x3',
        'conv_7x1_1x7',
        'dil_conv_3x3',
        'avg_pool_3x3',
        'max_pool_3x3',
        'max_pool_5x5',
        'max_pool_7x7',
        'conv_1x1',
        'conv_3x3',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
    ]
    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__(self.NASNET_OPS,
                         merge_op='loose_end',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)


@model_wrapper
class ENAS(NDS):
    __doc__ = """Search space proposed in `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~ENAS.ENAS_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    # ENAS_OPS = [
    #     'skip_connect',
    #     'sep_conv_3x3',
    #     'sep_conv_5x5',
    #     'avg_pool_3x3',
    #     'max_pool_3x3',
    # ]
    ENAS_OPS = [
        'none',
        'priv_max_pool_3x3',
        'priv_avg_pool_3x3',
        'priv_skip_connect',
        'priv_sep_conv_3x3_relu',
        'priv_sep_conv_3x3_selu',
        'priv_sep_conv_3x3_tanh',
        'priv_sep_conv_3x3_linear',
        'priv_sep_conv_3x3_htanh',
        'priv_sep_conv_3x3_sigmoid',
    ]
    """The candidate operations."""
#我们希望在子类中保留父类方法的行为，可以使用super()函数来调用父类的方法并在其基础上进行修改
    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        print("locate file")
        #在初始化过程中，它调用了父类NDS的__init__方法，并传递了一些参数，用于构建基本的模型架构搜索空间。
        super().__init__(self.ENAS_OPS,
                         merge_op='loose_end',#原本是loose_end
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)


@model_wrapper
class AmoebaNet(NDS):
    __doc__ = """Search space proposed in
    `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~AmoebaNet.AMOEBA_OPS`.
    It has 5 nodes per cell, and the output is concatenation of nodes not used as input to other nodes.
    """ + _INIT_PARAMETER_DOCS

    AMOEBA_OPS = [
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'avg_pool_3x3',
        'max_pool_3x3',
        'dil_sep_conv_3x3',
        'conv_7x1_1x7',
    ]
    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):

        super().__init__(self.AMOEBA_OPS,
                         merge_op='loose_end',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)


@model_wrapper
class PNAS(NDS):
    __doc__ = """Search space proposed in
    `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
    It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
    """ + _INIT_PARAMETER_DOCS

    PNAS_OPS = [
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'conv_7x1_1x7',
        'skip_connect',
        'avg_pool_3x3',
        'max_pool_3x3',
        'dil_conv_3x3',
    ]
    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__(self.PNAS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)


@model_wrapper
class DARTS(NDS):
    __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
    It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.

    .. note::

        ``none`` is not included in the operator candidates.
        It has already been handled in the differentiable implementation of cell.

    """ + _INIT_PARAMETER_DOCS

   # DARTS_OPS = [
   #     'none',
   #     'priv_max_pool_3x3',
   #     'priv_avg_pool_3x3',
   #     'priv_skip_connect',
   #     'priv_sep_conv_3x3_relu',
   #     'priv_sep_conv_3x3_selu',
   #     'priv_sep_conv_3x3_tanh',
   #     'priv_sep_conv_3x3_linear',
   #     'priv_sep_conv_3x3_htanh',
   #     'priv_sep_conv_3x3_sigmoid',
   #   ]
    DARTS_OPS = [
        'none',
        'priv_max_pool_3x3',
        'priv_avg_pool_3x3',
        'priv_skip_connect',
        'priv_sep_conv_3x3_relu',
        'priv_sep_conv_3x3_selu',
        'priv_sep_conv_3x3_tanh',
        'priv_sep_conv_3x3_linear',
        'priv_sep_conv_3x3_htanh',
        'priv_sep_conv_3x3_sigmoid',
     ]

    # DARTS_OPS = [
    #         'none',
    #         'priv_max_pool_3x3',
    #         'priv_avg_pool_3x3',
    #         'priv_skip_connect',
    #         'priv_sep_conv_3x3_tanh',
    #         'priv_sep_conv_3x3_softsign',
    #         'priv_sep_conv_3x3_htanh',
    #         'priv_sep_conv_3x3_sigmoid',
    # ]
    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        print("111112222111111")
        super().__init__(self.DARTS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=5,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = False, progress: bool = True
    ) -> nn.Module:

        init_kwargs = {}  # all default

        if name == 'darts-v2':
            init_kwargs.update(
                num_cells=20,
                width=36,
            )
            arch = {
                'normal/op_2_0': 'sep_conv_3x3',
                'normal/op_2_1': 'sep_conv_3x3',
                'normal/input_2_0': 0,
                'normal/input_2_1': 1,
                'normal/op_3_0': 'sep_conv_3x3',
                'normal/op_3_1': 'sep_conv_3x3',
                'normal/input_3_0': 0,
                'normal/input_3_1': 1,
                'normal/op_4_0': 'sep_conv_3x3',
                'normal/op_4_1': 'skip_connect',
                'normal/input_4_0': 1,
                'normal/input_4_1': 0,
                'normal/op_5_0': 'skip_connect',
                'normal/op_5_1': 'dil_conv_3x3',
                'normal/input_5_0': 0,
                'normal/input_5_1': 2,
                'reduce/op_2_0': 'max_pool_3x3',
                'reduce/op_2_1': 'max_pool_3x3',
                'reduce/input_2_0': 0,
                'reduce/input_2_1': 1,
                'reduce/op_3_0': 'skip_connect',
                'reduce/op_3_1': 'max_pool_3x3',
                'reduce/input_3_0': 2,
                'reduce/input_3_1': 1,
                'reduce/op_4_0': 'max_pool_3x3',
                'reduce/op_4_1': 'skip_connect',
                'reduce/input_4_0': 0,
                'reduce/input_4_1': 2,
                'reduce/op_5_0': 'skip_connect',
                'reduce/op_5_1': 'max_pool_3x3',
                'reduce/input_5_0': 2,
                'reduce/input_5_1': 1
            }

        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.fixed_arch(arch)
        model = model_factory(**init_kwargs)

        if pretrained:
            weight_file = load_pretrained_weight(name, download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model

