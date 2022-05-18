import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from neuromancer.dynamics import SSM 
from neuromancer import blocks
from neuromancer import integrators
import types

class GraphFeatures(nn.Module):
    def __init__(self, node_attr=None, edge_attr=None):
        """Tuple-like class that encapsulates node and edge attributes while overloading
        operators to apply to them individually.

        :param node_attr: [Tensor], defaults to None
        :param edge_attr: [Tensor], defaults to None
        """
        super(GraphFeatures, self).__init__()
        self.node_attr = node_attr
        self.edge_attr = edge_attr

    def __add__(self, y):
        if type(y) is GraphFeatures:
            return GraphFeatures(self.node_attr + y.node_attr,
                                 self.edge_attr + y.edge_attr)
        return GraphFeatures(self.node_attr + y, self.edge_attr + y)

    def __iadd__(self, y):
        if type(y) is GraphFeatures:
            self.node_attr += y.node_attr
            self.edge_attr += y.edge_attr
        else:
            self.node_attr += y
            self.edge_attr += y
        return self

    def __mul__(self, y):
        if type(y) is GraphFeatures:
            return GraphFeatures(self.node_attr * y.node_attr,
                                 self.edge_attr * y.edge_attr)
        return GraphFeatures(self.node_attr * y, self.edge_attr * y)

    def __rmul__(self,y):
        return self.__mul__(y)

    def __imul__(self,y):
        if type(y) is GraphFeatures:
            self.node_attr *= y.node_attr
            self.edge_attr *= y.edge_attr
        else:
            self.node_attr *= y
            self.edge_attr *= y
        return self

    def __truediv__(self,y):
        if type(y) is GraphFeatures:
            return GraphFeatures(self.node_attr / y.node_attr,
                                 self.edge_attr / y.edge_attr)
        return GraphFeatures(self.node_attr / y, self.edge_attr / y)

    def __rtruediv__(self,y):
        if type(y) is GraphFeatures:
            return GraphFeatures(y.node_attr /self.node_attr,
                                 y.edge_attr / self.edge_attr)
        return GraphFeatures(y / self.node_attr, y / self.edge_attr)

    def __getitem__(self, attr):
        if 'node' in attr:
            return self.node_attr
        elif 'edge' in attr:
            return self.edge_attr
        else:
            return None

    @staticmethod
    def cat(x, dim=0):
        node_attr = torch.cat([y.node_attr for y in x], dim)
        edge_attr = torch.cat([y.edge_attr for y in x], dim)
        return GraphFeatures(node_attr,edge_attr)

        
class GraphFeatureUpdate(nn.Module):
    def __init__(self, node_f=None, edge_f=None):
        """Module to update the node and edge attributes of GraphFeatures

        :param node_enc: _description_, defaults to None
        :param edge_enc: _description_, defaults to None
        """
        super(GraphFeatureUpdate, self).__init__()
        self.node_f = node_f if node_f is not None else lambda x : x
        self.edge_f = edge_f if edge_f is not None else lambda x : x
        self.in_features=["node_attr", "edge_attr"]
        self.out_features = ["node_attr","edge_attr"]
    
    def forward(self, g):
        """Calls node_f and edge_f on the node and edge attributes of a graph.

        :param g: [GraphFeatures]
        :return: [GraphFeatures] Modified node and edge attributes.
        """
        edge = self.edge_f(g.edge_attr)
        node = self.node_f(g.node_attr)
        return GraphFeatures(node, edge)
    
    def reg_error(self):
        return sum([k.reg_error() for k in [self.node_f, self.edge_f] if hasattr(k, "reg_error")])


class ArgSequential(nn.Sequential):
    def __init__(self, *args):
        """A version of a Sequential Module that shares additional args across modules.
        Used exactly like nn.Sequential but the forward function accepts additional arguments.
        Each module in the sequence must be able to accept the additional arguments as well.
        """
        super(ArgSequential, self).__init__(*args)
    
    def forward(self, X, *args, **kwargs):
        for module in self:
            X = module(X, *args, **kwargs)
        return X
    
    def reg_error(self):
        return sum([k.reg_error() for k in self if hasattr(k, "reg_error")])


class Replace(integrators.Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """An Integrator style module for x_t+1 = f(x_t)

        :param block: [nn.Module]
        :param interp_u: [function, nn.Module], defaults to None
        :param h: Not Used.
        """
        super(Replace, self).__init__(block, interp_u, h)

    def integrate(self, x, u, t):
        return self.block(self.state(x, t, t, u))
    

class nnLambda(nn.Module):
    def __init__(self, lam = lambda x: x):
        """nn.Module that encapsulates a lambda function.

        :param lam: [lambda], defaults to lambda x : x
        """
        super(nnLambda, self).__init__()
        assert type(lam) is types.LambdaType
        self.lam = lam
    
    def forward(self, *args):
        return self.lam(*args)
    

def graph_interp_u(tq, t, u):
    """Provides the interpetration to use integrators with graph features.
    Sums the edge attributes incident to a node and returns them as the node_attr.
    Concatenates the two node attributes incident to an edge and returns them as edge features.

    :param tq: Not used.
    :param t: Not used.
    :param u: {"edge_index": Tensor, "node_attr": Tensor, "edge_attr": Tensor}
    :return: [GraphFeatuers] 
    """
    row, col = u['edge_index'][0], u['edge_index'][1]
    node_attr = scatter_sum(u['edge_attr'], col, dim=0)
    edge_attr = torch.cat([u['node_attr'][row], u['node_attr'][col]], dim=-1)
    return GraphFeatures(node_attr, edge_attr)
