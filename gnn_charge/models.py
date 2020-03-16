# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp
import math
from dgl.nn import pytorch as dgl_pytorch

# =============================================================================
# MODULE CLASS
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
            self,
            in_feat,
            out_feat,
            model=dgl_pytorch.conv.SAGEConv,
            kwargs={'aggregator_type': 'mean'}):
        super(GN, self).__init__()
        self.gn = model(in_feat, out_feat, **kwargs)

    def forward(self, g):
        x = g.nodes['atom'].data['h']
        g_sub = dgl.to_homo(
            g.edge_type_subgraph(['atom_neighbors_atom']))
        x = self.gn(g_sub, x)
        g.nodes['atom'].data['h'] = x
        return g

class Net(torch.nn.Module):
    def __init__(self, config, readout_units=128, input_units=128):
        super(Net, self).__init__()

        dim = input_units
        self.exes = []

        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(117, input_units),
            torch.nn.Tanh())

        def apply_atom_in_graph(fn):
            def _fn(g):
                g.apply_nodes(
                    lambda node: {'h': fn(node.data['h'])}, ntype='atom')
                return g
            return _fn

        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except:
                pass

            if type(exe) == int:
                setattr(
                    self,
                    'd' + str(idx),
                    GN(dim, exe))

                dim = exe
                self.exes.append('d' + str(idx))

            elif type(exe) == str:
                activation = getattr(torch.nn.functional, exe)

                setattr(
                    self,
                    'a' + str(idx),
                    apply_atom_in_graph(activation))

                self.exes.append('a' + str(idx))

            elif type(exe) == float:
                dropout = torch.nn.Dropout(exe)
                setattr(
                    self,
                    'o' + str(idx),
                    apply_atom_in_graph(dropout))

                self.exes.append('o' + str(idx))

    def forward(self, g, return_graph=False):

        g.apply_nodes(
            lambda nodes: {'h': self.f_in(nodes.data['h0'])},
            ntype='atom')

        for exe in self.exes:
            g = getattr(self, exe)(g)

        return g
