"""
open-loop (directed acyclic graphs) and closed-loop (directed cyclic graphs) systems components

Minimum viable product
1, system class for open-loop rollout of autonomous nn.Module class
2, system class for open-loop rollout of non-autonomous nn.Module  class
3, system class for closed-loop rollout of simple DPC with neural policy and nonautonomous dynamics class (e.g. SSM, psl, ...)

Notes on simple implementation:

Time delay can be handled inside nodes simply or with more complexity
Sporadically sampled data can be handled prior with interpolation
Different time scales can be handled with nested systems
Networked systems seem like a natural fit here
"""
import os
import pydot
from itertools import combinations
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from neuromancer.dynamics.integrators import Euler
from neuromancer.modules import MLP
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.constraint import variable


class Node(nn.Module):
    """
    Simple class to handle cyclic computational graph connections. input_keys and output_keys
    define computational node connections through intermediate dictionaries.
    """
    def __init__(self, callable, input_keys, output_keys, name=None):
        """

        :param callable: Input: All input arguments are assumed to be torch.Tensors (batchsize, dim)
                         Output: All outputs are assumed to be torch.Tensors (batchsize, dim)
        :param input_keys: (list of str) For gathering inputs from intermediary data dictionary
        :param output_keys: (list of str) For sending inputs to other nodes through intermediary data dictionary
        :param name: (str) Unique node identifier
        """
        super().__init__()
        self.input_keys, self.output_keys = input_keys, output_keys
        self.callable, self.name = callable, name

    def forward(self, data):
        """
        This call function wraps the callable to receive/send dictionaries of Tensors

        :param datadict: (dict {str: 3-D Tensor (batch, time, dim)}) input to callable with associated input_keys
        :return: (dict {str: Tensor}) Output of callable with associated output_keys
        """
        inputs = [data[k] for k in self.input_keys]
        output = self.callable(*inputs)
        if not isinstance(output, tuple):
            output = [output]
        return {k: v for k, v in zip(self.output_keys, output)}

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"


class MovingHorizon(nn.Module):
    """
    The MovingHorizon class buffers single time step inputs for time-delay modeling from past ndelay
    steps. This class is a wrapper which does data handling for modules which take 3-d input (batch, time, dim)
    """
    def __init__(self, module, ndelay=1, history=None):
        """

        :param module: nn.Module which takes 3-d input dict and returns 2-d ouput dict
        :param ndelay: (int) Time-delay horizon
        :param history: (dict {str: list of Tensors}) An optional initialization of the history
                        buffer from previous measurements. There should be a key for each of
                        the input_keys in module and the values should be lists of 2-d tensors
        """
        super().__init__()
        self.input_keys, self.output_keys = module.input_keys, module.output_keys
        self.history = {k: [] for k in self.input_keys} if history is None else history
        self.ndelay, self.module = ndelay, module

    def forward(self, input):
        """
        The forward pass appends the input dictionary to the history buffer and gives
        last ndelay steps to the module. If history is blank the first step will be
        repeated ndelay times to initialize the buffer.

        :param input: (dict: str: 2-d tensor (batch, dim)) Dictionary of single step tensor inputs
        :return: (dict: str: 2-d Tensor (batch, dim)) Dictionary of single step tensor outputs
        """
        for k in self.input_keys:
            self.history[k].append(input[k])
            if len(self.history[k]) == 1:
                self.history[k] *= self.ndelay
        inputs = {k: torch.stack(self.history[k][-self.ndelay:]) for k in self.input_keys}
        return self.module(inputs)


class System(nn.Module):
    """
    Simple implementation for arbitrary cyclic computation
    """
    def __init__(self, nodes, name='', nstep_key='X', init_func=None, nsteps=None):
        """

        :param nodes: (list of Node objects)
        :param name: (str) Unique identifier for system class.
        :param nstep_key: (str) Key is used to infer number of rollout steps from input_data
        :param init_func: (callable(input_dict) -> input_dict) This function is used to set initial conditions of the system
        """
        super().__init__()
        self.nstep_key = nstep_key
        self.nsteps = nsteps
        self.nodes, self.name = nn.ModuleList(nodes), name
        if init_func is not None:
            self.init = init_func
        self.input_keys = set().union(*[c.input_keys for c in nodes])
        self.output_keys = set().union(*[c.output_keys for c in nodes])
        self.system_graph = self.graph()

    def graph(self):
        self._check_unique_names()
        graph = pydot.Dot("problem", graph_type="digraph", splines="spline", rankdir="LR")
        graph.add_node(pydot.Node("in", label="dataset", color='skyblue',
                                  style='filled', shape="box"))
        sim_loop = pydot.Cluster('sim_loop', color='cornsilk',
                                 style='filled', label='system')
        input_keys = []
        output_keys = []
        all_common_keys = []
        nonames = 1
        for node in self.nodes:
            input_keys += node.input_keys
            output_keys += node.output_keys
            if node.name is None:
                node.name = f'node_{nonames}'
                nonames += 1
            sim_loop.add_node(pydot.Node(node.name, label=node.name,
                                         color='lavender',
                                         style='filled',
                                         shape="box"))
        graph.add_node(pydot.Node('out', label='out', color='skyblue', style='filled', shape='box'))
        graph.add_subgraph(sim_loop)

        for src, dst in combinations(self.nodes, 2):
            common_keys = set(src.output_keys) & set(dst.input_keys)
            all_common_keys += common_keys
            for key in common_keys:
                graph.add_edge(pydot.Edge(src.name, dst.name, label=key))
            reverse_common_keys = set(dst.output_keys) & set(src.input_keys)
            for key in reverse_common_keys:
                graph.add_edge(pydot.Edge(dst.name, src.name, label=key))

        data_keys = list(set(input_keys) - set(all_common_keys))
        for node in self.nodes:
            loop_keys = list(set(node.input_keys) & set(node.output_keys))
            for key in loop_keys:
                graph.add_edge(pydot.Edge(node.name, node.name, label=key))
            for key in set(node.input_keys) & set(data_keys):
                graph.add_edge(pydot.Edge("in", node.name, label=key))
            for key in node.output_keys:
                graph.add_edge(pydot.Edge(node.name, 'out', label=key))
        self.input_keys = list(set(data_keys))
        self.output_keys = list(set(output_keys))
        return graph

    def show(self, figname=None):
        graph = self.system_graph
        if figname is not None:
            plot_func = {'svg': graph.write_svg,
                         'png': graph.write_png,
                         'jpg': graph.write_jpg}
            ext = figname.split('.')[-1]
            plot_func[ext](figname)
        else:
            graph.write_png('system_graph.png')
            img = mpimg.imread('system_graph.png')
            os.remove('system_graph.png')
            fig = plt.imshow(img, aspect='equal')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

    def _check_unique_names(self):
        num_unique = len([node.name for node in self.nodes])
        num_comp = len(self.nodes)
        assert num_unique == num_comp, \
            "All system nodes must have unique names " \
            "to construct a computational graph."

    def cat(self, data3d, data2d):
        """
        Concatenates data2d contents to corresponding entries in data3d
        :param data3d: (dict {str: Tensor}) Input to a node
        :param data2d: (dict {str: Tensor}) Output of a node
        :return:
        """
        for k in data2d:
            if k not in data3d:
                data3d[k] = data2d[k][:, None, :]
            else:
                data3d[k] = torch.cat([data3d[k], data2d[k][:, None, :]], dim=1)
        return data3d

    def init(self, data):
        """
        Any nodes in the graph that are start nodes will need some data initialized.
        Here is an example of initializing an x0 entry in the input_dict.

        Provide in base class analysis of computational graph. Label the source nodes. Keys for source nodes have to
        be in the data.
        """
        return data

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor}) Tensor shapes in dictionary are asssumed to be (batch, time, dim)
                                           If an init function should be written to assure that any 2-d or 1-d tensors
                                           have 3 dims.
        :return: (dict: {str: Tensor}) data with outputs of nstep rollout of Node interactions
        """

        nsteps = self.nsteps if self.nsteps is not None else data[self.nstep_key].shape[1]  # Infer number of rollout steps
        data = self.init(data)  # Set initial conditions of the system
        for i in range(nsteps):
            for node in self.nodes:
                indata = {k: data[k][:, i] for k in node.input_keys}  # collect what the compute node needs from data nodes
                outdata = node(indata)  # compute
                data = self.cat(data, outdata)  # feed the data nodes
        return data  # return recorded system measurements


if __name__ == "__main__":
    """
    Here is an example of a System that implements a standard n-step prediction rollout.
    This is the example illustrated in the slides.  
    """


    class MultipleShootingEuler(nn.Module):
        """
        Simple multiple shooting setup.
        """

        def __init__(self, nx, nu, hsize, nlayers, ts):
            super().__init__()
            self.dx = MLP(nx + nu, nx, bias=True, linear_map=nn.Linear, nonlin=nn.ELU,
                          hsizes=[hsize for h in range(nlayers)])
            interp_u = lambda tq, t, u: u
            self.integrator = Euler(self.dx, h=torch.tensor(ts), interp_u=interp_u)

        def forward(self, x1, xn, u):
            """

            :param x1: (Tensor, shape=(batchsize, nx))
            :param xn: (Tensor, shape=(batchsize, nx))
            :param u: (Tensor, shape=(batchsize, nu))
            :return: (tuple of Tensors, shapes=(batchsize, nx)) x2, xn+1
            """
            return self.integrator(x1, u=u), self.integrator(xn, u=u)


    m3 = EulerIntegrator(3, 2, 5, 2, 0.1)
    m4 = Node(m3, ['xn', 'U'], ['xn'])
    s2 = System([m4])
    # s2.show()
    s2.show('system_graph.png')
    exit()
    data = {'X': torch.randn(3, 2, 3), 'U': torch.randn(3, 2, 2)}
    print({k: v.shape for k, v in s2(data).items()})

    """
    This is an example of a System that implements a simple multiple shooting approach. 
    The MultipleShootingEuler module when wrapped as below will roll out for n-steps for a single initial 
    condition but also do simple 1-step ahead predictions. x1 draws an initial condition from the data whereas
    xn draws an initial condition from the output of the integrator.
    This allows you to optimize both n-step and 1-step rollout which has 
    proven to be effective for optimizing NODE surrogate models. 
    """
    m = MultipleShootingEuler(3, 2, 5, 2, 0.1)
    m2 = Node(m, ['X', 'xn', 'U'], ['xstep', 'xn'])
    s = System([m2])
    data = {'X': torch.randn(3, 2, 3), 'U': torch.randn(3, 2, 2)}
    print({k: v.shape for k, v in data.items()})
    data = s(data)
    print({k: v.shape for k, v in data.items()})
    """
    Test for compatibility with Problem
    """
    xpred = variable('xn')[:, :-1, :]
    xtrue = variable('X')
    loss = (xpred == xtrue)^2
    obj = PenaltyLoss([loss], [])
    p = Problem([s], obj)
    data['name'] = 'test'
    print(p(data))




