# python base imports
import os
import pydot
from itertools import combinations
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import warnings
import lightning.pytorch as pl 
from typing import Dict, List, Callable
from inspect import signature

# machine learning/data science imports
import torch
import torch.nn as nn



class LitProblem(pl.LightningModule):
    """
    A PyTorch-Lightning Module wrapper for the Neuromancer Problem class. 
    As is customary with LightningModules, steps for training and validation are outlined here, as well as the optimizer
    Logging metrics are also defined here, such as 'train_loss'. 
    """

    # Class attrinbute for expected signatures of Lightning hooks
    expected_signatures = {
        'backward': '(self, loss)',
        'on_before_backward': '(self, loss)',
        'on_after_backward': '(self)',
        'on_before_zero_grad': '(self, optimizer)',
        'on_fit_start': '(self)',
        'on_fit_end': '(self)', 
        'on_load_checkpoint': '(self, checkpoint)', 
        'on_save_checkpoint': '(self, checkpoint)', 
        'on_train_start': '(self)', 
        'on_train_end': '(self)', 
        'on_validation_start': '(self)', 
        'on_validation_end': '(self)', 
        'on_test_batch_start': '(self, batch, batch_idx, dataloader_idx)', 
        'on_test_batch_end': '(self, batch, batch_idx, dataloader_idx)', 
        'on_test_epoch_start': '(self)', 
        'on_test_epoch_end': '(self)', 
        'on_test_start': '(self)', 
        'on_test_end': '(self)', 
        'on_predict_batch_start': '(self, batch, batch_idx, dataloader_idx)', 
        'on_predict_batch_end': '(self, batch, batch_idx, dataloader_idx)', 
        'on_predict_epoch_start': '(self)', 
        'on_predict_epoch_end': '(self)', 
        'on_predict_start': '(self)', 
        'on_predict_end': '(self)', 
        'on_train_batch_start': '(self, batch, batch_idx)',
        'on_train_batch_end': '(self, batch, batch_idx)',
        'on_train_epoch_start': '(self)', 
        'on_train_epoch_end': '(self)', 
        'on_validation_batch_start': '(self, batch, batch_idx)',
        'on_validation_batch_end': '(self, batch, batch_idx)',
        'on_validation_epoch_start': '(self)', 
        'on_validation_epoch_end': '(self)', 
        'configure_model': '(self)'

    }

    def __init__(self, problem, train_metric='train_loss', dev_metric='train_loss', test_metric='train_loss', custom_optimizer=None, 
                 custom_training_step=None, custom_hooks=None, hparam_config=None):
        """
        :param problem: A Neuromancer Problem()
        :param train_metric: metric to be used during training step. Default to train_loss
        :param dev_metric: metric to be used during validation step. Default to train_loss
        :param test_metric: metric to be used during testing step (currently not supported yet)
        :param custom_optimizer: Optimizer to be used during training. Default is None, in which an 
                                 Adam optimizer is used with learning rate = 0.001
        :param custom_training_step: Custom training step function, if desired. Defaults to None, in which case the standard training step procedure is executed
        :param custom_hooks: Dictionary of custom hook functions that are supported by Lightning. Defaults to None.
        :param hparam_config: A wandb hyperparameter configuration file. Only used for hyperparameter tuning. 
        """
        super().__init__()
        self.problem = problem
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        self.custom_optimizer = custom_optimizer
        self.custom_training_step = custom_training_step
        self.custom_hooks = custom_hooks or {}
        self.hparam_config = hparam_config
        self.lr = .001

        self.training_step_outputs = []
        self.validation_step_outputs = []

        
        self._load_from_config()
        self._validate_hooks()

    def _load_from_config(self): 
        if self.hparam_config: 
            if "learning_rate" in self.hparam_config: 
                self.lr = self.hparam_config.learning_rate
    
    def _validate_hooks(self):
        for hook_name, hook_func in self.custom_hooks.items():
            if hook_name in self.expected_signatures:
                expected_sig = self.expected_signatures[hook_name]
                actual_sig = str(signature(hook_func))
                if actual_sig != expected_sig:
                    raise ValueError(f"Custom hook '{hook_name}' has incorrect signature: expected {expected_sig}, got {actual_sig}")
    
    def training_step(self, batch, batch_idx):
        if self.custom_training_step is not None: 
            loss = self.custom_training_step(self, batch)
        else: 
            output = self.problem(batch)
            loss = output[self.train_metric]
        self.training_step_outputs.append(loss)
        self.log('train_loss', loss, on_epoch=True, enable_graph=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if 'on_train_epoch_end' in self.custom_hooks:
            self.custom_hooks['on_train_epoch_end'](self)
        else:
            epoch_average = torch.stack(self.training_step_outputs).mean()
            self.log("training_epoch_average", epoch_average) #log to lightning_logs
            self.training_step_outputs.clear() 

    def validation_step(self, batch, batch_idx):
        if 'validation_step' in self.custom_hooks: 
            self.custom_hook['validation_step'](self, batch, batch_idx)
        else: 
            output = self.problem(batch)
            loss = output[self.dev_metric]
            self.validation_step_outputs.append(loss)
            self.log('dev_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        if 'configure_optimizers' in self.custom_hooks:
            self.custom_hooks['configure_optimizers'](self)
        else: 
            if self.custom_optimizer is None: 
                optimizer = torch.optim.Adam(self.problem.parameters(), self.lr, betas=(0.0, 0.9))
            else: 
                optimizer = self.custom_optimizer
            return optimizer
    
    def get_problem(self):
        return self.problem
    

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
        graph.add_node(pydot.Node("in", label="dataset", color='skyblue',
                                  style='filled', shape="box"))
        graph.add_node(pydot.Node("out", label="loss", color='lightcoral',
                                  style='filled', shape="box"))
        # plot clusters for nodes and loss terms
        node_cluster = pydot.Cluster('nodes', color='cornsilk',
                                 style='filled', label='nodes')
        obj_cluster = pydot.Cluster('loss_term', color='cornsilk',
                                 style='filled', label='loss terms')

        # create nodes in the node cluster
        input_keys = []
        output_keys = []
        nonames = 1
        for idx, node in enumerate(self.nodes):
            input_keys += node.input_keys
            output_keys += node.output_keys
            if node.name is None or node.name == '':
                node.name = f'node_{nonames}'
                nonames += 1
            node_cluster.add_node(pydot.Node(node.name, color='lavender', style='filled',
                                         label=node.name, shape="box"))
        graph.add_subgraph(node_cluster)

        # get keys of recurrent nodes
        loop_keys = []
        for node in self.nodes:
            loop_keys += set(node.input_keys) & set(node.output_keys)

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

        # get keys required as input and to initialize some nodes
        init_keys = set(input_keys) - (set(output_keys)-set(loop_keys))
        # get keys required as input to nodes from the dataset
        data_keys = set(input_keys)-set(output_keys)
        # create input connections to the dataset if not provided by previous node
        previous_output_keys = []
        for node in self.nodes:
            for key in set(node.input_keys) & (init_keys-set(previous_output_keys)):
                graph.add_edge(pydot.Edge("in", node.name, label=key))
            previous_output_keys += node.output_keys

        # add objectives and constraints in the graph
        if include_objectives:
            # get keys required as input to objectives from the dataset
            obj_input_keys = []
            for i, obj in enumerate(self.loss.objectives + self.loss.constraints):
                obj_input_keys += obj.input_keys
            obj_data_keys = set(obj_input_keys) - set(output_keys)
            # create connections
            for i, obj in enumerate(self.loss.objectives+self.loss.constraints):
                # choose different colors for objective terms and constraints
                if i+1 <= len(self.loss.objectives):
                    color = "lightpink"
                else:
                    color = 'thistle'
                # add loss term boxes
                obj_cluster.add_node(pydot.Node(obj.name, label=obj.name,
                                          shape="box", color=color, style='filled'))
                # connect nodes to loss terms
                unique_common_keys = set()
                for node in reverse_order_nodes:
                    common_keys = set(node.output_keys) & set(obj.input_keys)
                    for key in common_keys:
                        if key not in unique_common_keys:
                            graph.add_edge(pydot.Edge(node.name, obj.name, label=key))
                            unique_common_keys.add(key)
                # generate tuples connecting input data to loss terms
                for key in obj_data_keys:
                    if key in obj.input_keys:
                        graph.add_edge(pydot.Edge("in", obj.name, label=key))
                graph.add_edge(pydot.Edge(obj.name, "out", label=obj.name))
            graph.add_subgraph(obj_cluster)
        else:
            # aggregate outputs in a single output node
            for node in self.nodes:
                for key in set(node.output_keys) & set(self.loss.input_keys):
                    graph.add_edge(pydot.Edge("out", node.name, label=key))
            for key in data_keys & set(self.loss.input_keys):
                graph.add_edge(pydot.Edge("in", "out", label=key))

        input_keys += self.loss.input_keys
        self.input_keys = list(set(input_keys))
        output_keys += self.loss.output_keys
        self.output_keys = list(set(output_keys))
        return graph

    def show(self, figname=None):
        graph = self.graph()
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
            plt.figure()
            fig = plt.imshow(img, aspect='equal')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

    def freeze(self):
        """
        Freezes the parameters of all nodes in the system
        """
        for node in self.nodes:
            node.freeze()

    def unfreeze(self):
        """
        Unfreezes the parameters of all nodes in the system
        """
        for node in self.nodes:
            node.unfreeze()

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

