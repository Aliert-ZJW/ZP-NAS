import torch
import sys, os
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict

sys.path.append(os.path.dirname(sys.path[0]))

from train.search_param_cells import NASNetSearchCell as SearchCell

class NASNetworkDARTS(nn.Module):
    '''
      'name': 'DARTS-V1',
      'C': 1, 'N': 1, 'depth': 2, 'steps': 4,'use_stem': True,thin_use_stem:False,'multiplier': 4,
       'stem_multiplier': 1,
      'num_classes': class_num,  1000
      'space': search_space, ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']
      'affine': True, 'track_running_stats': bool(xargs.track_running_stats), True
      'super_type': xargs.super_type, nasnet-super
    '''
    def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
                 num_classes: int, index_space_normal, index_space_reduce, edge, drop_path_prob, affine: bool,
                 track_running_stats: bool, use_stem=True):
        super(NASNetworkDARTS, self).__init__()
        self._C = C
        self._layerN = N  # number of stacked cell at each stage
        self._steps = steps
        self._multiplier = multiplier  #  The number of channels multiplier factor because there are 4 intermediate nodes represents a 4-fold increase in the number of channels
        self.index_space_normal = index_space_normal
        self.index_space_reduce = index_space_reduce
        self.edge = edge
        self.use_stem = use_stem
        self.drop_path_prob = drop_path_prob
        self.stem = nn.Sequential(
           nn.Conv2d(3, stem_multiplier, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(stem_multiplier))
        # self.stem0 = nn.Sequential(
        #     nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(C // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(C),
        # )
        #
        # self.stem1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(C),
        # )

        # config for each layer
        layer_channels = [C] * N + [C*2] + [C*2] * N + [C*4] + [C*4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        num_edge, edge2index = None, None
        # Why is it still four times worse
        C_prev_prev, C_prev, C_curr, reduction_prev = stem_multiplier, stem_multiplier, C, False
        # C_prev_prev, C_prev, C_curr, reduction_prev = C, C, C, True

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(index_space_normal, index_space_reduce, edge, steps, multiplier,
                              C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier*C_curr, reduction
        self.op_names = deepcopy(index_space_normal[0])
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_normal_parameters = nn.Parameter(1e-3*torch.randn(num_edge, len(index_space_normal[0])))
        self.arch_reduce_parameters = nn.Parameter(1e-3*torch.randn(num_edge, len(index_space_normal[0])))

    def get_weights(self) -> List[torch.nn.Parameter]:
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist += list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
        xlist += list( self.classifier.parameters() )
        return xlist

    def get_alphas(self) -> List[torch.nn.Parameter]:
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def set_alphas(self, arch_parameters):
        self.arch_normal_parameters.data.copy_(arch_parameters[0].data)
        self.arch_reduce_parameters.data.copy_(arch_parameters[1].data)

    def show_alphas(self) -> Text:
        with torch.no_grad():
            A = 'arch-normal-parameters :\n{:}'.format( nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu() )
            B = 'arch-reduce-parameters :\n{:}'.format( nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu() )
        return '{:}\n{:}'.format(A, B)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self) -> Text:
        return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def genotype2cmd(self, genotype):
        cmd = "Genotype(normal=%s, normal_concat=[2, 3, 4, 5], reduce=%s, reduce_concat=[2, 3, 4, 5])"%(genotype['normal'], genotype['reduce'])
        return cmd

    def genotype(self) -> Dict[Text, List]:
        def _parse(weights):
            #  weights [14, 8]
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):   # [0, 1, 2, 3] 代表了中间节点
                end = start + n
                W = weights[start:end].copy()  # Gets the weight of the current intermediate node to the previous successor node
                selected_edges = []
                a = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.op_names.index('none')))
                _edge_indice = a[:2]
                for _edge_index in _edge_indice:
                    _op_indice = list(range(W.shape[1]))
                    _op_indice.remove(self.op_names.index('none'))
                    q = sorted(_op_indice, key=lambda x: -W[_edge_index][x])
                    _op_index = q[0]
                    selected_edges.append((self.op_names[_op_index], _edge_index))
                gene += selected_edges
                start = end
                n += 1
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return self.genotype2cmd({'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))})

    def forward(self, inputs):

        normal_w = nn.functional.softmax(self.arch_normal_parameters, dim=1)
        reduce_w = nn.functional.softmax(self.arch_reduce_parameters, dim=1)
        normal_a = self.arch_normal_parameters.detach().clone()
        reduce_a = self.arch_reduce_parameters.detach().clone()

        s0 = s1 = self.stem(inputs)
        # s0 = self.stem0(inputs)
        # s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                ww, aa = reduce_w, reduce_a
            else:
                ww, aa = normal_w, normal_a
            # s0, s1 = s1, cell.forward_darts(s0, s1, ww, aa, self.drop_path_prob)
            s0, s1 = s1, cell.forward_darts(s0, s1, ww, aa)
        # out = self.lastact(s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits, out