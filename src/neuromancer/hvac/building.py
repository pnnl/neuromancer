"""
This module provides a bridge between the BuildingComponent interface and the Node/System
architecture from NeuroMANCER, enabling building components to benefit from automatic wiring, trajectory
storage, and graph visualization for interpretable end-to-end building HVAC simulation.
One can either directly use the BuildingNode class as a drop in replacement for Nodes in
construction of a NeuroMANCER System or use this extension of the System class (BuildingSystem)
to take advantage of convenient defaults in the BuildingComponent interface.

This BuildingSystem class uses a two-stage initialization approach for continuous-time building systems.
First, context-driven initialization (see context.py and self.context in BuildingComponent classes)
ensures all components start from physically consistent conditions
(realistic temperatures, loads, and operating points) rather than arbitrary values, dramatically reducing startup transients.
Second, when needed for high-precision applications, laboratory-style warmup with fixed exogenous variables
(constant weather, setpoints, disturbances) drives the system to true thermodynamic equilibrium
where each component's steady-state output becomes the proper input for downstream components.
This combination eliminates both the startup transients of poorly initialized systems and
the temporal causality problem where components receive transient responses rather
than steady-state flows from upstream components. For 1R1C systems, this approach typically achieves convergence
within hours due to simple exponential thermal dynamics, providing mathematically rigorous initial conditions
for stable continuous-time simulation.

In contrast EnergyPlus's periodic warmup (repeating the first simulation day until convergence)
is well-suited for annual energy analysis, where buildings naturally operate in daily cycles
and the goal is understanding long-term performance under realistic diurnal patterns.
However, this approach is problematic for control policy learning applications
that often require short rollouts (minutes to hours), arbitrary time horizons not aligned with 24-hour cycles,
and reproducible initial conditions independent of calendar dates.
Control algorithms may need to start episodes at any time of day or weather condition,
making daily-periodic initialization irrelevant or counterproductive.
Our laboratory warmup with completely constant exogenous factors provides the precise,
reproducible steady-state baseline that control applications require:
every training episode begins from an identical thermodynamic equilibrium,
eliminating initialization variance that could interfere with policy learning.
This is especially critical for reinforcement learning, where policy gradients can be corrupted
by inconsistent starting conditions, and for systematic control algorithm comparisons
where initialization artifacts must not confound performance differences.
"""
import torch
from typing import Dict, List, Optional
from torch_buildings.building_components.base import BuildingComponent
# from neuromancer.system import Node, System

import os
import pydot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from neuromancer.constraint import Variable

class Node(nn.Module):
    """
    Simple class to handle cyclic computational graph connections. input_keys and output_keys
    define computational node connections through intermediate dictionaries.
    """
    def __init__(self, callable, input_keys, output_keys, name=None):
        """

        :param callable: Input: All input arguments are assumed to be torch.Tensors (batchsize, dim)
                         Output: All outputs are assumed to be torch.Tensors (batchsize, dim)
        :param input_keys: (list of str or Variable) For gathering inputs from intermediary data dictionary
        :param output_keys: (list of str or Variable) For sending inputs to other nodes through intermediary data dictionary
        :param name: (str) Unique node identifier
        """
        super().__init__()
        self.input_keys = [
            var.key if isinstance(var, Variable) else var for var in input_keys
        ]
        self.output_keys = [
            var.key if isinstance(var, Variable) else var for var in output_keys
        ]
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

    def freeze(self):
        """
        Freezes the parameters of the callable in this node
        """
        for param in self.callable.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the parameters of the callable in this node
        """
        for param in self.callable.parameters():
            param.requires_grad = True

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"


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

        # build I/O and node loop connections
        loop_keys = []
        init_keys = []
        previous_output_keys = []
        for idx_node, node in enumerate(self.nodes):
            node_loop_keys = set(node.input_keys) & set(node.output_keys)
            loop_keys += node_loop_keys
            init_keys += set(node.input_keys) - set(previous_output_keys)
            previous_output_keys += node.output_keys

            # build single node recurrent connections
            for key in node_loop_keys:
                graph.add_edge(pydot.Edge(node.name, node.name, label=key))
            # build connections to the dataset
            for key in set(node.input_keys) & set(init_keys):
                graph.add_edge(pydot.Edge("in", node.name, label=key))
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
        graph = self.graph()
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
                print({k: data[k].shape for k in node.input_keys})
                indata = {k: data[k][:, i] for k in node.input_keys}  # collect what the compute node needs from data nodes
                outdata = node(indata)  # compute
                data = self.cat(data, outdata)  # feed the data nodes
            print({k: v.shape for k, v in data.items()})

        return data  # return recorded system measurements

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


class BuildingNode(nn.Module):
    """
    Adapter class that wraps BuildingComponent instances to work with the System/Node architecture.
    Inherits the Node interface so that BuildingComponent instances can be used directly with the System class,
    or alternatively the BuildingSystem subclass of System.
    """
    def __init__(self,
                 component: BuildingComponent,
                 input_map: Dict[str, str] = None,
                 name: Optional[str] = None):
        """
        Initialize BuildingNode wrapper around a BuildingComponent.

        Args:
            component: Instance of BuildingComponent to wrap
            input_keys: List of input variable names.
            name: Node name for graph visualization
            suppress_warnings: If True, suppress warnings about missing input data
        """
        super().__init__()
        name = name or f"{type(component).__name__}_{id(component)}"
        self.input_keywords = [k for k in component._state_ranges] + [k for k in component._external_ranges]

        assert len(input_map) == len(self.input_keywords)
        self.input_map = input_map
        # Add time and time delta to input keys
        self.input_map.update({'t': 't', 'dt': 'dt'})
        input_keys = [k for k in self.input_map]
        input_keys.extend(['t', 'dt'])
        self.input_keys = input_keys
        self.output_keys = self._make_output_keys(component, name)
        self.component = component
        self.callable, self.name = component, name

    def _make_output_keys(self, component, name) -> List[str]:
        """Infer output keys from component's range specifications."""
        keys = []

        # Output variables
        if hasattr(component, '_output_ranges'):
            keys.extend(component._output_ranges.keys())

        # State variables (components output their updated states)
        if hasattr(component, '_state_ranges'):
            keys.extend(component._state_ranges.keys())
        keys = [f'{name}.{k}' for k in keys]
        return keys

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass that adapts between Node and BuildingComponent interfaces.

        Args:
            data: Dictionary containing all available variables

        Returns:
            Dictionary of all component outputs (not filtered by output_keys)
        """
        # Gather inputs for component from data dictionary and map to keyword arguments
        inputs = {self.input_map[k]: data[k] for k in self.input_keys}
        outputs = self.component(**inputs)
        # Append name to output keys
        return {f'{self.name}.{k}': v for k, v in outputs.items()}

    def __repr__(self):
        return f"BuildingNode({self.name}: {', '.join(self.input_keys)} -> {', '.join(self.output_keys)})"


class BuildingSystem(System):
    """
    Building-specific extension of the System class. Can serve as inspiration for a more
    general NetworkedSystem class.

    Inherits all functionality from System (rollout loop, graph visualization,
    automatic wiring, etc.) and adds building-specific conveniences:
    - Building-appropriate defaults (nstep_key, dt)
    - Automatic state/input initialization from BuildingComponent methods
    - Convenience simulation method for validation/data generation
    """

    def __init__(self,
                 nodes: List[BuildingNode],
                 **kwargs):
        """
        Args:
            nodes: List of BuildingNode or Node instances
            name: System name for identification
            **kwargs: Additional arguments passed to System.__init__()
        """
        kwargs.setdefault('nstep_key', 'dt')
        kwargs.setdefault('name', "BuildingSystem")
        super().__init__(nodes, **kwargs)

    def forward(self, input_dict, t_start: float = 0.0, dt=300.0, warmup=5):
        """
        Args:
            input_dict: Input data dictionary
            t_start: Start time [s] if 't' not in input_dict (default: 0.0)
            dt: Time step [s] if 'dt' not in input_dict (default: 300.0)
            warmup: (int) Number of warmup iterations to converge on initial inputs default=5

        Returns:
            Dictionary of trajectory results from System.forward()
        """
        data = input_dict.copy()
        data = self.setup(data=data, dt=dt)
        data = self._warmup_initialization(data, warmup)
        return super().forward(data)

    def _warmup_initialization(self, data, iterations):
        """
        Warmup will only execute nodes in the graph which are BuildingNodes.
        This avoids random behavior from Nodes which may be initialized in
        a machine learning setting such as a policy node which is an MLP with random initial weights.

        Args:
            data: Input data dictionary
            iterations: Number of warmup iterations to converge on initial inputs default=5
        :return: Input dictionary with initial inputs
        """
        with torch.no_grad():
            init_data = {k: v[:, 0] for k, v in data.items()}  # Extract initial guesses from data
            building_nodes = [node for node in self.nodes if isinstance(node, BuildingNode)]
            for i in range(iterations):
                # Run one timestep with current guesses
                out_data = {}
                for node in building_nodes:
                    out_data.update(node(init_data))
                init_data.update(out_data)
        for var in data:
            data[var][:, 0] = init_data[var]
        return data

    def setup(self, data, dt=300.0):
        """
        Initialize ALL component variables for proper time alignment.
        This solves the phenomena of off-by-dt for continuous time
        modeling of networked dynamical systems.
        """
        data = data.copy()

        # Infer simulation parameters
        batch_size = next(iter(data.values())).shape[0] if data else 1
        nsteps = data[self.nstep_key].shape[1] if self.nstep_key in data else 1

        # Add time parameters if not present
        if 't' not in data:
            times = torch.arange(nsteps, dtype=torch.float32) * dt
            data['t'] = times.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        else:
            t_start = data['t'][:, 0].item()

        if 'dt' not in data:
            data['dt'] = torch.full((batch_size, nsteps), dt).unsqueeze(-1)

        # Initialize missing input variables
        building_components = [n for n in self.nodes if isinstance(n, BuildingNode)]
        for node in building_components:
            for k in node.input_keys:
                if k not in data:
                    key = node.input_map[k]
                    if key in node.component._state_ranges:
                        data[k] = node.component.initial_state_functions()[key](batch_size).unsqueeze(1)
                    else:
                        data[k] = node.component.input_functions[key](t_start, batch_size).unsqueeze(1)
        return data

    def simulate(self,
                 duration_hours: float = 24.0,
                 dt_minutes: float = 5.0,
                 batch_size: int = 1,
                 external_inputs: Optional[Dict] = None,
                 t_start: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        High-level building simulation interface. Really just a wrapper around the forward pass
        of BuildingSystem for easy specification of the simulation window and sampling rate.

        Args:
            duration_hours: Simulation duration [hours]
            dt_minutes: Time step [minutes]
            batch_size: Number of parallel simulations
            external_inputs: Custom external input data (optional)
            t_start: Simulation start time [s]

        Returns:
            Dictionary of trajectory results
        """
        # Setup time parameters
        dt = dt_minutes * 60.0  # Convert to seconds
        nsteps = int(duration_hours * 3600 / dt)

        # Prepare input data dictionary
        data = {} if external_inputs is None else external_inputs.copy()

        # Add time trajectory for nstep inference
        times = torch.arange(nsteps, dtype=torch.float32) * dt + t_start
        data['t'] = times.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        data['dt'] = torch.full((batch_size, nsteps), dt).unsqueeze(-1)

        return self.forward(data, t_start=t_start, dt=dt)


