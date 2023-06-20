"""

"""
# python base imports
import os
import pydot
from itertools import combinations
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Callable

# machine learning/data science imports
import torch
import torch.nn as nn


class Problem(nn.Module):
    """
    This class is similar in spirit to a nn.Sequential module. However,
    by concatenating input and output dictionaries for each node
    module we can represent arbitrary directed acyclic computation graphs.
    In addition the Problem module takes care of calculating loss functions
    via given instantiated weighted multi-objective PenaltyLoss object which
    calculate objective and constraints terms from aggregated input and set
    of outputs from the node modules.
    """

    def __init__(self, nodes: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]],
                 loss: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
                 grad_inference=False, check_overwrite=False):
        """
        :param nodes: (List[Node]) list of objects which implement the Node interface
                                   (i.e. input and output are dicts of Tensors and
                                   object has input_keys, output_keys, and name attributes)
        :param loss: (PenaltyLoss) instantiated loss class
        :param update: (Callable) problem will update the output dictionary and return new dictionary with the same keys
                                but updated values. Example includes projected gradient method.
        :param grad_inference: (boolean) flag for enabling computation of grdients during inference time, useful for techniques like projected gradient
        """
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.loss = loss
        self.grad_inference = grad_inference
        self.check_overwrite = check_overwrite
        self._check_keys()
        self.problem_graph = self.graph()

    def _check_keys(self):
        keys = set()
        for node in list(self.nodes)+[self.loss]:
            keys |= set(node.input_keys)
            new_keys = set(node.output_keys)
            same = new_keys & keys
            if self.check_overwrite:
                if len(same) != 0:
                    warnings.warn(f'Keys {same} are being overwritten by the node {node}.')
            keys |= new_keys

    def _check_unique_names(self):
        num_unique = len(set([o.name for o in self.loss.objectives] + [c.name for c in self.loss.constraints]
                             + [comp.name for comp in self.nodes]))
        num_obj = len(self.loss.objectives) + len(self.loss.constraints) + len(self.nodes)
        assert num_unique == num_obj, \
            "All nodes, objectives and constraints must have unique names to construct a computational graph."

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.step(data)
        output_dict = self.loss(output_dict)
        if isinstance(output_dict, torch.Tensor):
            output_dict = {self.loss.name: output_dict}
        return {f'{data["name"]}_{k}': v for k, v in output_dict.items()}

    def step(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for node in self.nodes:
            output_dict = node(input_dict)
            if isinstance(output_dict, torch.Tensor):
                output_dict = {node.name: output_dict}
            input_dict = {**input_dict, **output_dict}
        return input_dict

    def graph(self, include_objectives=True):
        self._check_unique_names()
        graph = pydot.Dot("problem", graph_type="digraph", splines="spline", rankdir="LR")
        graph.add_node(pydot.Node("in", label="dataset", shape="box"))
        graph.add_node(pydot.Node("out", label="loss", shape="box"))
        input_keys = []
        output_keys = []
        all_common_keys = []
        nonames = 1
        for idx, node in enumerate(self.nodes):
            input_keys += node.input_keys
            output_keys += node.output_keys
            if node.name is None:
                node.name = f'node_{nonames}'
                nonames += 1
            graph.add_node(pydot.Node(node.name, label=node.name, shape="box"))

        for src, dst in combinations(self.nodes, 2):
            common_keys = set(src.output_keys) & set(dst.input_keys)
            all_common_keys += common_keys
            for key in common_keys:
                graph.add_edge(pydot.Edge(src.name, dst.name, label=key))

        data_keys = list(set(input_keys) - set(all_common_keys))
        for node in self.nodes:
            for key in set(node.input_keys) & set(data_keys):
                graph.add_edge(pydot.Edge("in", node.name, label=key))

        if include_objectives:
            _add_obj_nodes(graph, self.loss.objectives, self.nodes, data_keys)
            _add_obj_nodes(graph, self.loss.constraints, self.nodes, data_keys, style="dashed")
        else:
            for node in self.nodes:
                for key in set(node.output_keys) & set(self.loss.input_keys):
                    graph.add_edge(pydot.Edge("out", node.name, label=key))
                for key in set(data_keys) & set(self.loss.input_keys):
                    graph.add_edge(pydot.Edge("in", "out", label=key))

        input_keys += self.loss.input_keys
        self.input_keys = list(set(input_keys))
        output_keys += self.loss.output_keys
        self.output_keys = list(set(output_keys))
        return graph

    def show(self, figname=None):
        graph = self.problem_graph
        if figname is not None:
            plot_func = {'svg': graph.write_svg,
                         'png': graph.write_png,
                         'jpg': graph.write_jpg}
            ext = figname.split('.')[-1]
            plot_func[ext](figname)
        else:
            graph.write_png('problem_graph.png')
            img = mpimg.imread('problem_graph.png')
            os.remove('problem_graph.png')
            fig = plt.imshow(img, aspect='equal')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

    def __repr__(self):
        s = "### MODEL SUMMARY ###\n\nnodeS:"
        if len(self.nodes) > 0:
            for c in self.nodes:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nCONSTRAINTS:"
        if len(self.loss.constraints) > 0:
            for c in self.loss.constraints:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nOBJECTIVES:"
        if len(self.loss.objectives) > 0:
            for c in self.loss.objectives:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        return s


def _add_obj_nodes(graph, objs, nodes, data_keys, style="solid"):
    for obj in objs:
        graph.add_node(pydot.Node(obj.name, label=obj.name, shape="box", color="red", style=style))
        common_keys = [
            (c.name, k) for c in nodes for k in c.output_keys
            if k in obj.input_keys
        ] + [
            ("in", k) for k in data_keys
            if k in obj.input_keys
        ]
        for n, key in common_keys:
            graph.add_edge(pydot.Edge(n, obj.name, label=key))
        graph.add_edge(pydot.Edge(obj.name, "out", label=obj.name))
