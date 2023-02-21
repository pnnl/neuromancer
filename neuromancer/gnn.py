import torch
import torch.nn as nn
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
        self.out_features = ["node_attr", "edge_attr"]
    
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

class GraphFeatureBatchNorm(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
        self.NodeNorm = nn.BatchNorm1d(size)
        self.EdgeNorm = nn.BatchNorm1d(size)
        
    def forward(self, G : GraphFeatures):
        #TODO Figure out how we actually want to do this
        node_shape = G.node_attr.shape
        edge_shape = G.edge_attr.shape
        G.node_attr = self.NodeNorm(G.node_attr.view(-1, self.size)).view(node_shape)
        G.edge_attr = self.EdgeNorm(G.edge_attr.view(-1, self.size)).view(edge_shape)
        return G
    
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
    

class NodeDeltaPreprocessor(nn.Module):
    def __init__(self, attr_key, in_features, transform=None, symmetric=False) -> None:
        """
        Class to initialize edge attributes in a graph as 
        the difference between node features
        
        :param attr_key: Name of node attributes in the input dict
        :param transform: nn.Module to transform node attributes
        :param symmetric: If true, will take the absolute value of the difference
        """
        super().__init__()
        self.key=attr_key
        self.in_features=in_features
        self.transform = transform
        self.symmetric = symmetric

        
    def forward(self, data):
        """
        :param data: dict containing "key" corresponding to node attr, opt: edge_index, batch
        :returns: {'node_attr':tensor, 'edge_attr':tensor, 'edge_index':tensor}
        """
        node_attr = data[self.key] #(batch, ?, nodestates)
        if node_attr.ndim==3:
            node_attr = node_attr[:,0]
        batch, nodestates = node_attr.shape
        nodes = nodestates // self.in_features
        node_attr = node_attr.reshape(batch, nodes, self.in_features)
            
        if self.transform:
            node_attr = self.transform(node_attr)
        
        edge_index = data.get('edge_index',
                              self.make_edge_index(nodes).to(node_attr.device))
        src, dst = edge_index[0], edge_index[1]
        edge_attr = node_attr[:,dst,:] - node_attr[:,src,:]
        if self.symmetric:
            edge_attr = torch.abs(edge_attr)
        return {'node_attr': node_attr,
                'edge_attr': edge_attr,
                'edge_index': edge_index}
    
    def make_edge_index(self, nodes):
        src = torch.arange(nodes,dtype=torch.long).repeat_interleave(nodes)
        dst = torch.arange(nodes,dtype=torch.long).repeat(nodes)          
        edge_index = torch.vstack([src,dst])
        return edge_index
        
class RCUpdate(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        edge_index = edge_index.long()
        self.src, self.dst = edge_index[0], edge_index[1]

        self.edges = edge_index.shape[1]
        self.nodes = torch.max(edge_index).item()+1
        
        self.R = nn.Parameter(
            torch.Tensor(size=(1, self.edges, 1)).uniform_(4,10)
        )
        self.C = nn.Parameter(
            torch.Tensor(size=(1, self.nodes, 1)).uniform_(0.0001,0.002)
        )
        
        #To symmeterize R we need to find the locations of every pair of flipped edges
        edge_map = {(i.item(),j.item()) : idx for idx, (i,j) in enumerate(zip(self.src,self.dst))}
        self.edge_map = torch.tensor(
            [edge_map[(d.item(), s.item())] for (s,d) in zip(self.src,self.dst)],
            dtype=torch.long)
        self.in_features, self.out_features = 1, 1
        
    def forward(self, G, *args):
        T = G.node_attr
        #Temperature Delta
        # (1/Ci*R_ij) * (Ti-Tj)
        edge_attr = (self.R + self.R[:, self.edge_map]) * self.C[:, self.src] * (T[:, self.dst] - T[:, self.src])
        
        #Node Update sums deltas
        idx= self.dst.reshape(1, -1, 1)
        idx = idx.repeat(edge_attr.shape[0], 1, edge_attr.shape[-1])
        node_delta = torch.zeros_like(T)
        node_delta.scatter_reduce_(dim=1, index=idx, src=edge_attr, reduce='sum')
        return GraphFeatures(node_delta, edge_attr)

class RCPreprocessor(nn.Module):
    def __init__(self, edge_index, nodes=None, input_key_map={}) -> None:
        super().__init__()
        self.edge_index=edge_index.long()
        self.nodes = nodes if nodes is not None else torch.max(edge_index).item()+1
        self.input_key_map=input_key_map

        
    def forward(self, data):
        #(batch, time, node*states) --> (batch, time, nodes, states)
        node_attr = data[self.input_key_map['x0']]
        if node_attr.ndim==3:
            node_attr = node_attr[:, 0]
        batch, nodestates = node_attr.shape
        states = nodestates//self.nodes
        node_attr = node_attr.reshape(batch, self.nodes, states)
        
        edge_attr = node_attr[:, self.edge_index[0], :] - node_attr[:, self.edge_index[1], :]
        out = {"node_attr": node_attr, 
               "edge_attr": edge_attr, 
               "edge_index": self.edge_index}
        return out
        
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
    edge_attr = torch.cat([u['node_attr'][...,row,:], u['node_attr'][...,col,:]], dim=-1)
    
    col = col.to(u['edge_attr'].device).reshape(1,-1,1)
    col = col.repeat(u['edge_attr'].shape[0], 1, u['edge_attr'].shape[-1])
    node_attr = torch.zeros_like(u['node_attr'])
    node_attr.scatter_reduce_(dim=1, index=col, src=u['edge_attr'], reduce='sum')

    return GraphFeatures(node_attr, edge_attr)
