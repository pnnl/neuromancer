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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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

        :param datadict: (dict {str: Tensor}) input to callable with associated input_keys
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
        :return: (dict: str: 3-d Tensor (ndelay, batch, dim)) Dictionary of tensor outputs for the last ndelay times
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
    def __init__(self, nodes, name=None, nstep_key='X', init_func=None, nsteps=None):
        """

        :param nodes: (list of Node objects)
        :param name: (str) Unique identifier for system class.
        :param nstep_key: (str) Key is used to infer number of rollout steps from input_data
        :param init_func: (callable(input_dict) -> input_dict) This function is used to set initial conditions of the system
        :param nsteps: (int) prediction horizon (rollout steps) length
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
        nonames = 1
        for node in self.nodes:
            input_keys += node.input_keys
            output_keys += node.output_keys
            if node.name is None or node.name == '':
                node.name = f'node_{nonames}'
                nonames += 1
            sim_loop.add_node(pydot.Node(node.name, label=node.name,
                                         color='lavender',
                                         style='filled',
                                         shape="box"))
        graph.add_node(pydot.Node('out', label='out', color='skyblue', style='filled', shape='box'))
        graph.add_subgraph(sim_loop)

        # build node connections in reverse order
        reverse_order_nodes = self.nodes[::-1]
        for idx_dst, dst in enumerate(reverse_order_nodes):
            src_nodes = reverse_order_nodes[1+idx_dst:]
            unique_common_keys = set()
            for idx_src, src in enumerate(src_nodes):
                common_keys = set(src.output_keys) & set(dst.input_keys)
                for key in common_keys:
                    if key not in unique_common_keys:
                        graph.add_edge(pydot.Edge(src.name, dst.name, label=key))
                        unique_common_keys.add(key)

        # get keys of recurrent nodes
        loop_keys = []
        for node in self.nodes:
            node_loop_keys = set(node.input_keys) & set(node.output_keys)
            loop_keys += node_loop_keys
        # get keys required as input and to initialize some nodes
        init_keys = set(input_keys) - (set(output_keys) - set(loop_keys))

        # build I/O and node loop connections
        previous_output_keys = []
        for idx_node, node in enumerate(self.nodes):
            # build single node recurrent connections
            node_loop_keys = list(set(node.input_keys) & set(node.output_keys))
            for key in node_loop_keys:
                graph.add_edge(pydot.Edge(node.name, node.name, label=key))
            # build connections to the dataset
            for key in set(node.input_keys) & set(init_keys-set(previous_output_keys)):
                graph.add_edge(pydot.Edge("in", node.name, label=key))
            previous_output_keys += node.output_keys
            # build feedback connections for init nodes
            feedback_src_nodes = reverse_order_nodes[:-1-idx_node]
            if len(set(node.input_keys) & set(loop_keys) & set(init_keys)) > 0:
                for key in node.input_keys:
                    for src in feedback_src_nodes:
                        if key in src.output_keys and key not in previous_output_keys:
                            graph.add_edge(pydot.Edge(src.name, node.name, label=key))
                            break
        # build connections to the output of the system in a reversed order
        previous_output_keys = []
        for node in self.nodes[::-1]:
            for key in (set(node.output_keys) - set(previous_output_keys)):
                graph.add_edge(pydot.Edge(node.name, 'out', label=key))
            previous_output_keys += node.output_keys

        self.input_keys = list(set(init_keys))
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
            plt.figure()
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
        :return: (dict: {str: Tensor})
        """
        for k in data2d:
            if k not in data3d:
                data3d[k] = data2d[k][:, None, :]
            else:
                data3d[k] = torch.cat([data3d[k], data2d[k][:, None, :]], dim=1)
        return data3d

    def init(self, data):
        """
        :param data: (dict: {str: Tensor}) Tensor shapes in dictionary are asssumed to be (batch, time, dim)
        :return: (dict: {str: Tensor})

        Any nodes in the graph that are start nodes will need some data initialized.
        Here is an example of initializing an x0 entry in the input_dict.

        Provide in base class analysis of computational graph. Label the source nodes. Keys for source nodes have to
        be in the data.
        """
        return data

    def forward(self, input_dict):
        """

        :param input_dict: (dict: {str: Tensor}) Tensor shapes in dictionary are asssumed to be (batch, time, dim)
                                           If an init function should be written to assure that any 2-d or 1-d tensors
                                           have 3 dims.
        :return: (dict: {str: Tensor}) data with outputs of nstep rollout of Node interactions
        """
        data = input_dict.copy()
        nsteps = self.nsteps if self.nsteps is not None else data[self.nstep_key].shape[1]  # Infer number of rollout steps
        data = self.init(data)  # Set initial conditions of the system
        for i in range(nsteps):
            for node in self.nodes:
                indata = {k: data[k][:, i] for k in node.input_keys}  # collect what the compute node needs from data nodes
                outdata = node(indata)  # compute
                data = self.cat(data, outdata)  # feed the data nodes
        return data  # return recorded system measurements






