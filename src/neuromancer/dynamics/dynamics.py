"""
State space models (SSMs) for dynamical modeling.

Nomenclature:
    + x: states
    + y: predicted outputs
    + u: control inputs
    + d: uncontrolled inputs (measured disturbances)

Unstructured (blackbox) dynamical models:
    + :math:`x_{t+1} = f(x_t,u_t,d_t) \odot f_e(x_t)`
    + :math:`y_{t} =  f_y(x_t)`
    + :math:`\odot` is some operator acting on elements, e.g. + or *

Block-structured dynamical models:
    + :math:`x_{t+1} = f_x(x_t) \odot f_u(u_t) \odot f_d(d_t) \odot f_e(x_t)`
    + :math:`y_t =  f_y(x_t)`

Autonomous ordinary differential equation model:
    + :math:`x_{t+1} = ODESolve(f_x(x_t))`
    + :math:`y_t =  f_y(x_t)`

Non-autonomous ordinary differential equation model:
    + :math:`x_{t+1} = ODESolve(f_x(x_t, u_t, Time))`
    + :math:`y_t =  f_y(x_t)`

Graph-timestepper models:
    + :math: 'x_{t+1} = ODESolve(f_x(x_t, G, u_t, Time))
    + :math:`y_t =  f_y(x_t)`

Block components:
    + :math:`f_e` is an error model
    + :math:`f_{xudy}` is a nominal model
    + :math:`f_x` is the main state transition dynamics model
    + :math:`f_y` is the measurement function mapping latent dynamics to observations
    + :math:`f_u` are input dynamics
    + :math:`f_d` are disturbance dynamics
    + :math:`f_{yu}` models more direct effects of input on observations
    + :math:`ODESolve(f_x(x_t))` integrates the ODE system via solver, e.g. RK
"""

import torch
import torch.nn as nn
from typing import List
import neuromancer.slim as slim
from neuromancer.component import Component
from neuromancer import gnn
from neuromancer.blocks import MLP
from neuromancer import integrators

class SSM(Component):
    DEFAULT_INPUT_KEYS: List[str]
    DEFAULT_OUTPUT_KEYS: List[str]
    """
    Base State Space Model (SSM) class

    This class is used to manage naming of component input and output variables as they flow
    through the computational graph, as well as handle potential remapping of input and output keys
    to different names. It additionally provides a useful reference for users to see how components
    can be connected together in the overall computational graph.

    Components that inherit from this class should specify the class attributes DEFAULT_INPUT_KEYS
    and DEFAULT_OUTPUT_KEYS; these are used as the "canonical" names for input and output variables,
    respectively. These can be used to compare different components' output and input keys to see
    whether one component can accept another's output by default.

    By default, components have a `name` argument which is used to tag the output variables they
    generate; for instance, for a component called "estim", the canonical output "x0" is renamed to
    "x0_estim".

    >>> estim = LinearEstimator(..., name="estim")  # output "x0" remapped to "x0_estim"
    >>> ssm = BlockSSM(..., input_key_map={f"x0_{estim.name}": "x0"})  # input_keys used to remap to canonical name
    """

    def __init__(self, input_key_map={}, name=None):
        """
        :param input_key_map: (dict {str: str}) For renaming canonical input keys to state space model
        :param name: (str) Name will be appended to output keys.
        """
        self.update_input_keys(input_key_map=input_key_map)
        output_keys = [f"{k}_{name}" if name is not None else k for k in self.DEFAULT_OUTPUT_KEYS]
        super().__init__(input_keys=self.input_keys, output_keys=output_keys, name=name)

    def update_input_keys(self, input_key_map={}):
        assert isinstance(input_key_map, dict), \
            f"{type(self).__name__} input_key_map must be dict for remapping input variable names; "
        self.input_key_map = {
            **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_key_map.keys()},
            **input_key_map
        }
        self.input_keys = list(self.input_key_map.values())


class BlockSSM(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]
    """
    Block-structured system dynamical models:
        :math:`x_{t+1} = f_x(x_t) \odot f_u(u_t) \odot f_d(d_t) \odot f_e(x_t)`
        :math:`y_t =  f_y(x_t)`
    """
    def __init__(self, fx, fy, fu=None, fd=None, fe=None, fyu=None,
                 xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add,
                 residual=False, name='block_ssm', input_key_map={}):
        """
        :param fx: (nn.Module) State transition function
        :param fy: (nn.Module) Observation function
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param fe: (nn.Module) Error term via state augmentation
        :param fyu: (nn.Module) Input-Observation function
        :param xou: (callable) Elementwise tensor op
        :param xod: (callable) Elementwise tensor op
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        """
        if fu is not None:
            self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + ['Uf']
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fU']
        if fd is not None:
            self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + ['Df']
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fD']
        if fe is not None:
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fE']

        super().__init__(input_key_map, name)

        self.fx, self.fy, self.fu, self.fd, self.fe, self.fyu = fx, fy, fu, fd, fe, fyu
        self.nx, self.ny, self.nu, self.nd = (
            self.fx.in_features,
            self.fy.out_features,
            self.fu.in_features if fu is not None else 0,
            self.fd.in_features if fd is not None else 0
        )

        self.check_features()
        self.residual = residual

        self.xou, self.xod, self.xoe, self.xoyu = xou, xod, xoe, xoyu

    def check_features(self):
        self.nx, self.ny = self.fx.in_features, self.fy.out_features

        assert self.fx.in_features == self.fx.out_features, \
            'State transition must have same input and output dimensions'
        assert self.fy.in_features == self.fx.out_features, \
            'Output map must have same input size as output size of state transition'
        if self.fu is not None:
            assert self.fu.out_features == self.fx.out_features, \
                'Dimension mismatch between input and state transition'
        if self.fd is not None:
            assert self.fd.out_features == self.fx.out_features, \
                'Dimension mismatch between disturbance and state transition'

    def forward(self, data):
        """
        performs nstep ahead rollout of a given dynamical system model
        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]
        X, Y, FD, FU, FE = [], [], [], [], []
        x = data[self.input_key_map['x0']]
        for i in range(nsteps):
            x_prev = x
            x = self.fx(x)
            if self.fu is not None:
                fu = self.fu(data[self.input_key_map['Uf']][:, i, :])
                x = self.xou(x, fu)
                FU.append(fu)
            if self.fd is not None:
                fd = self.fd(data[self.input_key_map['Df']][:, i, :])
                x = self.xod(x, fd)
                FD.append(fd)
            if self.fe is not None:
                fe = self.fe(x_prev)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(data[self.input_key_map['Uf']][:, i, :])
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)
        tensor_lists = [X, Y, FU, FD, FE]
        tensor_lists = [t for t in tensor_lists if t]
        output = {name: torch.stack(tensor_list, dim=1) for tensor_list, name
                  in zip(tensor_lists, self.output_keys[1:])
                  if tensor_list}
        output[self.output_keys[0]] = self.reg_error()
        return output

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class LinearSSM(BlockSSM):
    """
    Implementation of discrete-time Linear State Space Model (LSSM).
        x_{t+1} = A x_t + B u_t + E d_t
        y_{t+1} = C x_{t+1}
    Where:
        x - state variables
        y - measurement variables
        y - control input variables
        d - disturbance variables

    LinearSSM is implemented as a specific instance of BlockSSM.

    for more information about LSSMs see:
        https://en.wikipedia.org/wiki/State-space_representation
        https://www.mathworks.com/help/ident/ug/what-are-state-space-models.html
    """
    def __init__(self, A, B, C, E=None, input_key_map={}, name=None):
        """
        :param A: (torch.tensor) state/system matrix
        :param B: (torch.tensor) input matrix
        :param C: (torch.tensor) output matrix
        :param E: (torch.tensor) disturbance matrix
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param name: (str) Name for tracking output
        """
        assert isinstance(A, torch.Tensor), \
            f'State matrix A must be torch.Tensor type, ' \
            f'got {type(A)}'
        assert isinstance(B, torch.Tensor), \
            f'Input matrix B must be torch.Tensor type, ' \
            f'got {type(B)}'
        assert isinstance(C, torch.Tensor), \
            f'Output matrix C must be torch.Tensor type, ' \
            f'got {type(C)}'
        assert len(A.shape) == 2, \
            f'State matrix A must be two dimensional, got {A.shape}'
        assert len(B.shape) == 2, \
            f'Input matrix B must be two dimensional, got {B.shape}'
        assert len(C.shape) == 2, \
            f'Output matrix C must be two dimensional, got {C.shape}'
        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]
        fu = slim.maps['linear'](nu, nx)
        fx = slim.maps['linear'](nx, nx)
        fy = slim.maps['linear'](nx, ny)
        fx.linear.weight = torch.nn.Parameter(A)
        fu.linear.weight = torch.nn.Parameter(B)
        fy.linear.weight = torch.nn.Parameter(C)
        if E is not None:
            assert isinstance(E, torch.Tensor), \
                f'Disturbance matrix E must be torch.Tensor type, ' \
                f'got {type(E)}'
            assert len(E.shape) == 2, \
                f'Disturbance matrix E must be two dimensional, got {E.shape}'
            nd = E.shape[1]
            fd = slim.maps['linear'](nd, nx)
            fd.linear.weight = torch.nn.Parameter(E)
        else:
            fd = None
        super().__init__(fx=fx, fy=fy, fu=fu, fd=fd, input_key_map=input_key_map, name=name)


class BlackSSM(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]
    """
    Black box state space model with unstructured system dynamics:
        :math:`x_{t+1} = f(x_t,u_t,d_t) \odot f_e(x_t)`
        :math:`y_{t} =  f_y(x_t)`
        :math:`\odot` is some operator acting on elements, e.g. + or *
    """

    def __init__(self, fx, fy, fe=None, fyu=None, xoe=torch.add, xoyu=torch.add, name='black_ssm',
                 input_key_map={}, extra_inputs=[]):
        """
        :param fx: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param fyu: (nn.Module) Input-Observation function
        :param fe: (nn.Module) Error term via state augmentation
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param extra_inputs: (list of str) Input keys to be added to canonical input.
        """
        self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + extra_inputs
        self.extra_inputs = extra_inputs
        if fe is not None:
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fE']

        super().__init__(input_key_map, name)

        self.fx, self.fy, self.fe, self.fyu = fx, fy, fe, fyu
        self.nx, self.ny = self.fx.out_features, self.fy.out_features
        self.xoe = xoe
        self.xoyu = xoyu

    def forward(self, data):
        """
        performs nstep ahead rollout of a given dynamical system model
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]
        X, Y, FE = [], [], []

        x = data[self.input_key_map['x0']]
        for i in range(nsteps):
            x_prev = x
            xplus = torch.cat([x] + [data[self.input_key_map[k]][:, i, :] for k in self.extra_inputs], dim=1)
            x = self.fx(xplus)
            if self.fe is not None:
                fe = self.fe(x_prev)
                x = self.xoe(x, fe)
                FE.append(fe)
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(xplus)
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)
        tensor_lists = [X, Y, FE]
        tensor_lists = [t for t in tensor_lists if t]
        output = {name: torch.stack(tensor_list, dim=1) for tensor_list, name
                  in zip(tensor_lists, self.output_keys[1:])
                  if tensor_list}
        output[self.output_keys[0]] = self.reg_error()
        return output

    def reg_error(self):
        """

        :return: 0-dimensional torch.Tensor
        """
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])

    def check_features(self):
        self.nx, self.ny = self.fx.out_features, self.fy.out_features
        assert self.fx.out_features == self.fy.in_features, 'Output map must have same input size as output size of state transition'


class ODEAuto(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]
    """
    State space model for solving autonomous ODEs w/ single-step integrators:
        :math:`x_{t+1} = ODESolve(f_x(x_t))`
        :math:`y_t =  f_y(x_t)`
        :math:`dx_t/dt = f_x(x_t)` - defined ODE system
        :math:`ODESolve(f_x(x_t))` - ODE solver that integrates the ODE system, e.g. RK
    """

    def __init__(self, fx, fy, name='dynamics', input_key_map={}):
        """
        :param fx: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        """
        self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS

        super().__init__(input_key_map, name)

        self.fx, self.fy = fx, fy
        self.nx, self.ny = self.fx.out_features, self.fy.out_features

    def forward(self, data):
        """
        performs nstep ahead rollout of a given dynamical system model
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]
        x = data[self.input_key_map['x0']]
        X, Y = [], []
        for i in range(nsteps):
            x = self.fx(x)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        tensor_lists = [X, Y]
        tensor_lists = [t for t in tensor_lists if t]
        output = {name: torch.stack(tensor_list, dim=1) for tensor_list, name
                  in zip(tensor_lists, self.output_keys[1:])
                  if tensor_list}
        output[self.output_keys[0]] = self.reg_error()
        return output

    def reg_error(self):
        """
        :return: 0-dimensional torch.Tensor
        """
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class ODENonAuto(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]
    """
    State space model for non-autonomous ODEs w/ single-step integrators:
        :math:`x_{t+1} = ODESolve(f_x(x_t, u_t, Time))`
        :math:`y_t =  f_y(x_t)`
        :math:`dx_t/dt = f_x(x_t, u_t, Time)` - defined ODE system with inputs u_t and Time
        :math:`ODESolve(f_x(x_t, u_t, Time))` - ODE solver that integrates the ODE system, e.g. RK
    """
    def __init__(self, fx, fy, name='dynamics', input_key_map={}, extra_inputs=[], online_flag=False):
        """
        :param fx: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param extra_inputs: (list of str) Input keys to be added to canonical input.
        :param online_flag: (bool) whether to use online interpolation or not.
        """
        self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + extra_inputs
        self.extra_inputs = extra_inputs
        super().__init__(input_key_map, name)
        self.fx, self.fy = fx, fy
        self.nx, self.ny = self.fx.out_features, self.fy.out_features
        self.online_flag = online_flag

    def forward(self, data):
        """
        performs nstep ahead rollout of a given dynamical system model
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]
        x = data[self.input_key_map['x0']]
        if 'Time' in self.input_key_map.keys():
            Time = data[self.input_key_map['Time']]  # (# of batches, nsteps, 1)
        else:
            # dummy time tensor that won't be used
            Time = torch.ones(data[self.input_key_map['Yf']].shape[0], nsteps, 1)
        inputs = torch.cat([data[self.input_key_map[k]] for k in self.extra_inputs], dim=-1) \
            if len(self.extra_inputs) is not 0 else Time
        X, Y = [], []
        for i in range(nsteps):
            if self.online_flag:
                if i == nsteps-1:
                    pass
                else:
                    x = self.fx(x, inputs[:, i:i+2, :], Time[:, i:i+2, :])
            else:
                x = self.fx(x, inputs[:, i, :], Time[:, i, :])
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        tensor_lists = [X, Y]
        tensor_lists = [t for t in tensor_lists if t]
        output = {name: torch.stack(tensor_list, dim=1) for tensor_list, name
                  in zip(tensor_lists, self.output_keys[1:])
                  if tensor_list}
        output[self.output_keys[0]] = self.reg_error()
        return output

    def reg_error(self):
        """

        :return: 0-dimensional torch.Tensor
        """
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class GraphTimestepper(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]

    def __init__(self,
                 num_nodes,
                 preprocessor=None,
                 graph_encoder=None,
                 processor=None,
                 decoder=None,
                 fu=None,
                 fd=None,
                 add_batch_norm = False,
                 latent_dim = None,
                 input_key_map={},
                 name="graph_timestepper",
                 device='cpu',):
        """Graph Neural Timestepper composed of a preprocessor, a encoder, processor, and decoder block.
        :param num_nodes: [int] Number of nodes in the input graph
        :param preprocessor: [nn.Module] Accepts a dictionary and returns a dictionary of tensors with keys [edge_index, node_attr, edge_attr, (opt) batch]. Other key/values will be added to output dict.
        :param graph_encoder: [nn.Module] Input Tensor: (nodes, num_node_features). Output Tensor: (nodes, latent_dim). Defaults to a linear layer.
        :param processor: [nn.ModuleList] Module to process the graph at each timestep
        :param decoder: [nn.Module] Input Tensor: (nodes, latent_dim). Output Tensor: (nodes, out_size)
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param add_batch_norm: [bool] If true, batch normalize node and edge features at each message passing step
        :param latent_dim: [int] Latent feature dimension requried if add_batch_norm is True
        :param input_key_map: [dict] Remaps the name of the inputs
        :param name: [string] Name of module. Prepended to all outputs.
        :param device: [torch.device], Device to run the model on, defaults to 'cpu'
        """
        if fu is not None:
            self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + ['Uf']
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fU']
        if fd is not None:
            self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + ['Df']
            self.DEFAULT_OUTPUT_KEYS = self.DEFAULT_OUTPUT_KEYS + ['fD']
        super().__init__(input_key_map, name)

        self.nodes = num_nodes
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        self.encoder = graph_encoder if graph_encoder is not None else nn.Identity()
        self.processor = processor if processor is not None else nn.ModuleList()
        self.decoder = decoder if decoder is not None else nn.Identity()
        self.latent_dim = latent_dim
        
        self.nx = self.nodes * self.latent_dim
        self.ny = self.nodes * (self.latent_dim if decoder is None else decoder.out_features)
        self.fu, self.fd = fu, fd
        self.nu = self.fu.in_features if fu is not None else 0
        self.nd = self.fd.in_features if fd is not None else 0
                
        self.batch_norm_processor = add_batch_norm
        if self.batch_norm_processor:
            assert self.latent_dim is not None
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.processor)):
                self.batch_norms.append(gnn.GraphFeatureBatchNorm(latent_dim))
        
        self.device = device
        self.to(self.device)

    def forward(self, data):
        """Forward pass of the module

        :param data: {"x0": Tensor, "Yf": Tensor}]
        :return: {"reg_error": Tensor, "X_pred": Tensor, "Y_pred": Tensor}
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]
        Uf = data.get(self.input_key_map.get('Uf','Uf'))
        Df = data.get(self.input_key_map.get('Df','Df'))

        X = self.preprocessor(data)
        edge_index = X.pop('edge_index')
        node_attr = X.pop('node_attr')
        edge_attr = X.pop('edge_attr')
        edges = edge_index.shape[1]
        if node_attr.ndim==2:
            # (batch, nodes*states) --> (batch, nodes, states)
            batch_size, nodestates = node_attr.shape
            nodes = self.nodes if self.nodes is not None else nodestates//self.latent_dim
            latent_dim = self.latent_dim if self.latent_dim is not None else nodestates//self.nodes
            node_attr = node_attr.reshape(batch_size, nodes, latent_dim)
            edge_attr = edge_attr.reshape(batch_size, edges, latent_dim)
        else:
            batch_size, nodes, latent_dim = node_attr.shape
        
        #Encode
        G = gnn.GraphFeatures(node_attr, edge_attr)
        G = self.encode(G)

        #Process
        X = [G]
        FU, FD = [], []
        for h in range(nsteps):
            G, fu, fd = self.process(G, edge_index, h, Uf, Df)
            X.append(G)
            if fu is not None: FU.append(fu)
            if fd is not None: FD.append(fd)

        #Decode
        Y=[]
        for x in X[1:]:
            y = self.decode(x)
            Y.append(y.reshape(batch_size, -1))

        #Y=torch.stack(Y, dim=1)
        X=[x['node_attr'].reshape(batch_size, -1) for x in X]
        #X=X.reshape(X.shape[0],X.shape[1],-1)

        tensor_lists = [X, Y, FU, FD]
        tensor_lists = [t for t in tensor_lists if t]
        out = {name: torch.stack(tensor_list, dim=1) for tensor_list, name
            in zip(tensor_lists, self.output_keys[1:])
            if tensor_list}
        out[self.output_keys[0]] = self.reg_error()
        return out

    def encode(self, G):
        """Encodes graph features into the latent space

        :param G: [Graph_Features]
        :return: [Graph_Features]
        """
        return self.encoder(G)

    def process(self, G, edge_index, t, U=None, D=None):
        """Processes a series of message passing steps on the graph.

        :param G: [Graph_Features]
        :param edge_index: [Tensor] Shape: (2, edges)
        :param t: [int] Current time
        :param U: [Tensor] Inputs
        :param D: [Tensor] Disturbances
        :return: [Graph_Features]
        """
        fu, fd = None, None
        for mps in range(len(self.processor)):
            u = {'node_attr':G['node_attr'], 'edge_attr':G['edge_attr'], 'edge_index':edge_index}
            if self.batch_norm_processor:
                G_next = self.processor[mps](self.batch_norms[mps](G), u)
            else:
                G_next = self.processor[mps](G, u)  
        if self.fu:
            fu = self.fu(G, U[:,t]).view(G.node_attr.shape)
            G_next.node_attr += fu
        if self.fd:
            fd = self.fd(G, D[:,t]).view(G.node_attr.shape)
            G_next.node_attr += fd

        return G_next, fu, fd

    def decode(self, G):
        """Transforms the graph features from the latent space to the output space. Drops edge features.

        :param G: [Graph_Features]
        :return: [Graph_Features]
        """
        return self.decoder(G['node_attr'])

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class MLPGraphTimestepper(GraphTimestepper):
    def __init__(self,
                 num_nodes,
                 num_node_features,
                 num_edge_features,
                 out_size,
                 latent_dim=128,
                 hidden_sizes=[128, 128],
                 message_passing_steps=2,
                 preprocessor=None,
                 graph_encoder=None,
                 integrator=None,
                 decoder=None,
                 fu=None, 
                 fd=None, 
                 add_batch_norm = True,
                 input_key_map={},
                 name="mlpg_timestepper",
                 device='cpu',
                 **kwargs):
        """Graph Neural Timestepper using MLP message passing layers.
        :param num_nodes: [int] Number of nodes in the graph
        :param num_node_features: [int] Number of node features after preprocesser
        :param num_edge_features: [int] Number of edge features after preprocesser
        :param out_size: [int] Dimension of output
        :param latent_dim: [int], Dimension of latent dim per node/edge, defaults to 128
        :param hidden_sizes: [list int], Dimensions of hidden sizes of nodes/edges, defaults to [128]
        :param message_passing_steps: [int], Number of message passing layers, defaults to 10
        :param preprocessor: [nn.Module] Accepts a dictionary and returns a dictionary of tensors with keys [edge_index, node_attr, edge_attr, (opt) batch]. Other key/values will be added to output dict.
        :param graph_encoder: [nn.Module] Input Tensor: (nodes, num_node_features). Output Tensor: (nodes, latent_dim). Defaults to a linear layer.
        :param integrator: [neuromancer.integrator] Integration scheme.
        :param decoder: [nn.Module] Input Tensor: (nodes, latent_dim). Output Tensor: (nodes, out_size)
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param separate_batch_dim: [bool] True when data should be handled with separate batch and node dimensions
        :param add_batch_norm: [bool] If true, performs batch normalization on node and edge features each step message passing step.
        :param input_key_map: [dict] Remaps the name of the inputs
        :param name: [string] Name of module. Prepended to all outputs.
        :param device: [torch.device], Device to run the model on, defaults to 'cpu'
        """
        self.node_features = num_node_features
        self.edge_features = num_edge_features
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes
        self.message_passing_steps = message_passing_steps
        self.out_size = out_size
        #Build default graph_encoder consisting of MLPs on node and edge features
        if graph_encoder is None:
            node_encoder = nn.Sequential(
                MLP(
                    insize=self.node_features,
                    outsize=self.latent_dim,
                    nonlin=nn.ReLU,
                    hsizes=self.hidden_sizes),
                nn.LayerNorm(self.latent_dim))
            edge_encoder = nn.Sequential(
                MLP(
                    insize=self.edge_features,
                    outsize=self.latent_dim,
                    nonlin=nn.ReLU,
                    hsizes=self.hidden_sizes),
                nn.LayerNorm(self.latent_dim))
            graph_encoder = gnn.GraphFeatureUpdate(node_encoder, edge_encoder)
        
        #Build Message Passing Processor composed of MLP layers
        processor = nn.ModuleList()
        integrator = integrators.integrators.get(integrator, gnn.Replace)
        for _ in range(self.message_passing_steps):
            node_process = nn.Sequential(
                MLP(
                    insize = self.latent_dim * 2,
                    outsize = self.latent_dim,
                    nonlin = nn.ReLU,
                    hsizes = self.hidden_sizes,
                ),
                nn.LayerNorm(self.latent_dim)
            )
            edge_process = nn.Sequential(
                MLP(
                    insize = self.latent_dim * 3,
                    outsize = self.latent_dim,
                    nonlin = nn.ReLU,
                    hsizes = self.hidden_sizes
                ),
                nn.LayerNorm(self.latent_dim)
            )
            block = integrator(
                        gnn.GraphFeatureUpdate(node_process, edge_process),
                        interp_u=gnn.graph_interp_u,
                        h=0.1)
            block.state = lambda x, tq, t, u: gnn.GraphFeatures.cat([x, block.interp_u(tq, t, u)], dim=-1)               
            processor.append(block)
    
        #Build the decoder
        decoder = decoder if decoder is not None else\
                        MLP(
                            insize=self.latent_dim,
                            outsize=self.out_size,
                            nonlin=nn.ReLU,
                            hsizes=self.hidden_sizes)
        super().__init__(
            num_nodes = num_nodes,
            preprocessor = preprocessor,
            graph_encoder = graph_encoder,
            processor = processor,
            decoder = decoder,
            fu = fu,
            fd = fd,
            add_batch_norm = add_batch_norm,
            latent_dim = latent_dim,
            input_key_map = input_key_map,
            name = name,
            device = device,
            **kwargs)
   

class RCTimestepper(GraphTimestepper):
    def __init__(self,
                 edge_index,
                 node_dim = 1,
                 nodes = None,
                 message_passing_steps=1,
                 preprocessor='default',
                 graph_encoder=None,
                 decoder=None,
                 integrator='Euler',
                 fu=None,
                 fd=None,
                 input_key_map={},
                 name="rc_timestepper",
                 device='cpu',):
        """
        Graph Timestepper using RCNetwork message passing layers.
        :param edge_index: [Tensor] graph adjacencies (2, num_edges)
        :param node_dim: [int] Number of features per node
        :param nodes: [int] Number of nodes in the graph
        :param message_passing_steps: [int] Number of iterations of message passing
        :param preprocessor: [nn.Module] Module to preprocess inputs
        :param graph_encoder: [nn.Module] Encode graph features from input space to latent space
        :param decoder: [nn.Module] Module to translate from latent space to output space
        :param integrator: [neuromancer.integrator] Integration scheme.
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param input_key_map: [dict] Dictionary remapping input_keys
        :param name: [str] Module name
        :param device: [torch.device] cpu/cuda:0/etc.
        """
        self.edge_index = edge_index
        nodes = nodes if nodes is not None else torch.max(self.edge_index).item() + 1
        
        #Build Default PreProcessor
        if preprocessor == 'default':
            preprocessor = gnn.RCPreprocessor(self.edge_index, input_key_map=input_key_map)
            
        #Build Processor
        processor = nn.ModuleList()
        integrator = integrators.integrators.get(integrator, gnn.Replace)
        for mp in range(message_passing_steps):
            processor.append(integrator(gnn.RCUpdate(self.edge_index)))
            
        super().__init__(num_nodes = nodes,
                 preprocessor = preprocessor,
                 graph_encoder = graph_encoder,
                 processor = processor,
                 decoder = decoder,
                 latent_dim=node_dim,
                 fu = fu,
                 fd = fd,
                 input_key_map = input_key_map,
                 name = name,
                 device = device)
        


def _extract_dims(datadims):
    xkey, ykey, ukey, dkey = ["x0", "Yf", "Uf", "Df"]
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0

    return nx, ny, nu, nd


def block_model(kind, datadims, linmap, nonlinmap, bias, n_layers=2, fe=None, fyu=None,
              activation=nn.GELU, residual=False, linargs=dict(),
              xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add, name='blockmodel', input_key_map={}):
    """
    Helper function that generates a block-structured SSM with the same structure used across fx, fy, fu, and fd.
    """
    assert kind in _bssm_kinds, \
        f"Unrecognized model kind {kind}; supported models are {_bssm_kinds}"

    nx, ny, nu, nd = _extract_dims(datadims)
    hsizes = [nx] * n_layers

    lin = lambda ni, no: (
        linmap(ni, no, bias=bias, linargs=linargs)
    )
    nlin = lambda ni, no: (
        nonlinmap(ni, no, bias=bias, hsizes=hsizes, linear_map=linmap, nonlin=activation, linargs=linargs)
    )

    if kind == "blocknlin":
        fx = nlin(nx, nx)
        fy = lin(nx, ny)
        fu = nlin(nu, nx) if nu != 0 else None
        fd = nlin(nd, nx) if nd != 0 else None
    elif kind == "linear":
        fx = lin(nx, nx)
        fy = lin(nx, ny)
        fu = lin(nu, nx) if nu != 0 else None
        fd = lin(nd, nx) if nd != 0 else None
    elif kind == "hammerstein":
        fx = lin(nx, nx)
        fy = lin(nx, ny)
        fu = nlin(nu, nx) if nu != 0 else None
        fd = nlin(nd, nx) if nd != 0 else None
    elif kind == "weiner":
        fx = lin(nx, nx)
        fy = nlin(nx, ny)
        fu = lin(nu, nx) if nu != 0 else None
        fd = lin(nd, nx) if nd != 0 else None
    else:   # hw
        fx = lin(nx, nx)
        fy = nlin(nx, ny)
        fu = nlin(nu, nx) if nu != 0 else None
        fd = nlin(nd, nx) if nd != 0 else None

    fe = (
        fe(nx, nx, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict())
        if kind in {"blocknlin", "hammerstein", "hw"}
        else fe(nx, nx, bias=bias, linargs=linargs)
    ) if fe is not None else None

    fyu = (
        fyu(nu, ny, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict())
        if kind in {"blocknlin", "hw"}
        else fyu(nu, ny, bias=bias, linargs=linargs)
    ) if fyu is not None else None

    model = BlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, fyu=fyu, xoyu=xoyu, xou=xou, xod=xod, xoe=xoe, name=name,
                 input_key_map=input_key_map, residual=residual)

    return model


def blackbox_model(datadims, linmap, nonlinmap, bias, n_layers=2, fe=None, fyu=None,
             activation=nn.GELU, linargs=dict(),
             xoyu=torch.add, xoe=torch.add, input_key_map={}, name='blackbox_model', extra_inputs=[]):
    """
    Helper function that generates a black box state space model.
    """
    nx, ny, nu, nd = _extract_dims(datadims)
    hsizes = [nx] * n_layers

    fx = nonlinmap(nx + nu + nd, nx, hsizes=hsizes,
                     bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs)
    fe = nonlinmap(nx, nx, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs) if fe is not None else None
    fyu = fyu(nu, ny, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs) if fyu is not None else None
    fy = linmap(nx, ny, bias=bias, linargs=linargs)

    model = BlackSSM(fx, fy, fe=fe, fyu=fyu, xoyu=xoyu, xoe=xoe, name=name,
                     input_key_map=input_key_map, extra_inputs=extra_inputs)
    return model


ssm_models_atoms = [BlockSSM, BlackSSM]
ssm_models_train = [block_model, blackbox_model]
_bssm_kinds = {
    "linear",
    "hammerstein",
    "wiener",
    "hw",
    "blocknlin"
}
