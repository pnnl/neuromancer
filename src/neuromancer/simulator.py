"""
Classes for construction of open-loop and closed-loop Simulators:
#.  SystemSimulator: aggregator class creating directed (possibly cyclic) computational graph
#.  SystemComponent: base abstract class for simulation components
"""

from typing import Dict, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import warnings
import pydot
from itertools import combinations
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class SystemComponent(ABC):
    """
    Base abstract class representing component of the Simulator system.
    The behavior of each component is defined in the _forward_step() method.
    Each component has atributes .input_keys and .output_keys given as list of strings
    that define input and ouptut variables of the component, respectively.
    """

    DEFAULT_INPUT_KEYS: List[str]
    OPTIONAL_INPUT_KEYS: List[str]
    DEFAULT_OUTPUT_KEYS: List[str]

    def __init__(self, name=None):
        """
        :param name: (str) Name for tracking output
        """
        super().__init__()
        self.name = name
        self.input_keys = self.DEFAULT_INPUT_KEYS
        self.output_keys = [f"{k}_{name}" if name is not None
                            else k for k in self.DEFAULT_OUTPUT_KEYS]

    def _check_inputs(self, data: Dict[str, np.ndarray]):
        set_diff = set(self.input_keys) - set(data.keys())
        keys = [k for k in data.keys()]
        assert len(set_diff) == 0, \
            f" Missing inputs {set_diff} only got {keys}"

    def _check_outputs(self, data: Dict[str, np.ndarray]):
        set_diff = set(self.output_keys) - set(data.keys())
        keys = [k for k in data.keys()]
        assert len(set_diff) == 0, \
            f" Missing outputs {set_diff} only got {keys}"

    @abstractmethod
    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        this internal method implements forward pass of a given system
        """
        pass

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        method for evaluating single step of a given system
        """
        self._check_inputs(data)
        data = self._forward_step(data)
        self._check_outputs(data)
        return data


class SystemSimulator():
    """
    SystemSimulator aggregates connected SystemComponent classes to form a simulation loop
    and performs simulation via .simulate() method.
    SystemSimulator supports simulation of directed cyclic computational graphs
    connecting SystemComponent via their input_keys and output_keys.
    """
    def __init__(self, components: List[SystemComponent]):
        """
        :param components: (List[SystemComponent]) list of objects which implement the SystemComponent interface (e.g. Dynamics, Policy, Estimator)
        """
        self.components = components
        self._check_keys()
        self._check_components()
        self.problem_graph = self.graph()
        keyList = list(set(self.input_keys).union(set(self.output_keys))) + ['timestep']
        self.history = {key: [] for key in keyList}

    def _check_keys(self):
        keys = set()
        for component in list(self.components):
            keys |= set(component.input_keys)
            new_keys = set(component.output_keys)
            same = new_keys & keys
            # if len(same) != 0:
            #     warnings.warn(f'Keys {same} are being overwritten by the component {component}.')
            keys |= new_keys

    def _check_components(self):
        for component in list(self.components):
            assert issubclass(type(component), SystemComponent), \
                f'Each closed loop component must be subclass of SystemComponent ' \
                f'got {type(component)}'

    def select_data_step(self, step_k, data_init: Dict[str, np.ndarray] = {},
                         data_traj: Dict[str, np.ndarray] = {}) -> Dict[str, np.ndarray]:
        data_traj = {k: v[step_k, :] for k, v in data_traj.items()}
        data = {**data_init, **data_traj}
        return data

    def simulate(self, nsim=1, data_init: Dict[str, np.ndarray] = {},
                 data_traj: Dict[str, np.ndarray] = {}) -> Dict[str, np.ndarray]:
        data = self.select_data_step(0, data_init, data_traj)
        self._log_step(data, 0)
        for k in range(1, nsim + 1):
            data_step = self.step(data)
            data = self.select_data_step(k, data_step, data_traj)
            self._log_step(data, k)
        trajectories = self.get_trajectories()
        return trajectories

    def step(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for component in self.components:
            output_dict = component.step(input_dict)
            if isinstance(output_dict, np.ndarray):
                output_dict = {component.name: output_dict}
            input_dict = {**input_dict, **output_dict}
        return input_dict

    def _log_step(self, data, step_k):
        self.history['timestep'] += [step_k]
        for key, value in data.items():
            self.history[key] += [value]

    def get_trajectories(self):
        trajectories = {k: np.stack(v, axis=0) for k, v in self.history.items()}
        return trajectories

    def graph(self):
        self._check_unique_names()
        graph = pydot.Dot("problem", graph_type="digraph", splines="spline", rankdir="LR")
        graph.add_node(pydot.Node("in", label="dataset", color='skyblue',
                            style='filled', shape="box"))
        sim_loop = pydot.Cluster('sim_loop', color='cornsilk',
                            style='filled', label='simulator loop')
        sim_loop.add_node(pydot.Node("time", label="time", color='wheat',
                            style='filled', shape="box"))
        sim_loop.add_edge(pydot.Edge("time", "time", label=''))
        input_keys = []
        output_keys = []
        all_common_keys = []
        nonames = 1
        for component in self.components:
            input_keys += component.input_keys
            output_keys += component.output_keys
            if component.name is None:
                component.name = f'comp_{nonames}'
                nonames += 1
            sim_loop.add_node(pydot.Node(component.name, label=component.name,
                                         color='lavender',
                                         style='filled',
                                         shape="box"))
        graph.add_subgraph(sim_loop)
        for src, dst in combinations(self.components, 2):
            common_keys = set(src.output_keys) & set(dst.input_keys)
            all_common_keys += common_keys
            for key in common_keys:
                graph.add_edge(pydot.Edge(src.name, dst.name, label=key))
        data_keys = list(set(input_keys) - set(all_common_keys))
        for component in self.components:
            loop_keys = list(set(component.input_keys) & set(component.output_keys))
            for key in loop_keys:
                graph.add_edge(pydot.Edge(component.name, component.name, label=key))
            for key in set(component.input_keys) & set(data_keys):
                graph.add_edge(pydot.Edge("in", component.name, label=key))
        self.input_keys = list(set(data_keys))
        self.output_keys = list(set(output_keys))
        return graph

    def _check_unique_names(self):
        num_unique = len([comp.name for comp in self.components])
        num_comp = len(self.components)
        assert num_unique == num_comp, \
            "All closed loop components must have unique names " \
            "to construct a computational graph."

    def plot_graph(self, fname='closed_loop_graph.png'):
        graph = self.problem_graph
        graph.write_png(fname)
        img = mpimg.imread(fname)
        fig = plt.imshow(img, aspect='equal')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()


class SystemDynamics(SystemComponent):
    DEFAULT_INPUT_KEYS = ["x"]
    OPTIONAL_INPUT_KEYS = ["u", "d", "w", "Time"]
    DEFAULT_OUTPUT_KEYS = ["x", "y"]

    """
    Base SystemComponent class for the dynamics models.
    
    Default input keys: 
        "x" - state variables
    Optional input keys:
        "u" - control input variables
        "d" - observable disturbance variables
        "w" - unobservable disturbance variables
        "Time" - time variable
    Default output keys:
        "x" - state variables
        "y" - observable state variables
        
    The forward step behavior of the model is defined via .eval_model() method.
    Forward step updates output key variables variables based on input key variables.
    """

    def __init__(self, input_key_map={}, name=None):
        """
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param name: (str) Name for tracking output
        """
        super().__init__(name=name)
        self.update_input_keys(input_key_map=input_key_map)

    def update_input_keys(self, input_key_map={}):
        assert isinstance(input_key_map, dict), \
            f"{type(self).__name__} input_key_map must be dict for remapping input variable names; "
        setdiff = set(input_key_map.keys()) - set(self.DEFAULT_INPUT_KEYS + self.OPTIONAL_INPUT_KEYS)
        assert len(setdiff) == 0, \
            f"Keys in input_key_map {input_key_map.keys()} must be in " \
            f"default keys {self.DEFAULT_INPUT_KEYS} or " \
            f"optional keys {self.OPTIONAL_INPUT_KEYS} "
        self.input_key_map = {
            **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_key_map.keys()},
            **input_key_map
        }
        self.input_keys = list(self.input_key_map.values())
        assert len(self.DEFAULT_INPUT_KEYS) <= len(self.input_keys) <= \
               len(self.DEFAULT_INPUT_KEYS) + len(self.OPTIONAL_INPUT_KEYS), \
            "Length of given input keys must be greater or equal than the " \
            "length of default input keys and must be " \
            "less or equal the length of default plus optional input keys"

    def _check_input_dims(self, data: Dict[str, np.ndarray]):
        for key in self.input_key_map.keys():
            key_shape = data[self.input_key_map[key]].shape
            assert len(key_shape) == 1, \
                f'Input {key} dimension must be 1, got {key_shape}'

    @abstractmethod
    def eval_model(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        this internal method implements forward pass of a given system
        """
        pass

    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self._check_input_dims(data)
        output = self.eval_model(data)
        return output


class DynamicsLinSSM(SystemDynamics):
    DEFAULT_INPUT_KEYS = ["x", "u"]
    OPTIONAL_INPUT_KEYS = ["d", "w", "Time"]
    DEFAULT_OUTPUT_KEYS = ["x", "y"]
    """
    Class implementing Linear Time Invariant State Space Model 
        x_{t+1} = A x_t + B u_t + E d_t + w_t
        y_{t+1} = C x_{t+1} + D u_t

    """

    def __init__(self, A, B, C, D=None, E=None, input_key_map={}, name=None):
        """
        :param A: (np.array) state dynamics matrix
        :param B: (np.array) input dynamics matrix
        :param C: (np.array) state to output matrix
        :param D: (np.array) input to output matrix
        :param E: (np.array) disturbance dynamics matrix
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param name: (str) Name for tracking output
        """
        super().__init__(input_key_map=input_key_map, name=name)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]

    def eval_model(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        x0 = data[self.input_key_map['x']]
        u = data[self.input_key_map['u']]
        x = np.matmul(self.A, x0) + np.matmul(self.B, u)
        if 'd' in self.input_key_map.keys() and self.E is not None:
            d = data[self.input_key_map['d']]
            x += np.matmul(self.E, d)
        if 'w' in self.input_key_map.keys():
            w = data[self.input_key_map['w']]
            x += w
        y = np.matmul(self.C, x)
        if self.D is not None:
            y += np.matmul(self.D, u)
        output = {}
        output[self.output_keys[0]] = x
        output[self.output_keys[1]] = y
        return output


class DynamicsPSL(SystemDynamics):
    """
    Wrapper for PSL system model simulating ordinary differential equation models:
        x_{t+1}, y_{t+1} = ODE(x_t, u_t, d_t)
    """

    def __init__(self, model, input_key_map={}, name=None):
        """
        :param model: (callable) instantiated PSL system model class
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param extra_inputs: (list of str) Input keys to be added to canonical input.
        :param name: (str) Name for tracking output
        """
        super().__init__(input_key_map=input_key_map, name=name)
        self.model = model
        self.nx = model.nx

    def eval_model(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        x0 = data[self.input_key_map['x']]
        if 'u' in self.input_key_map.keys():
            u = data[self.input_key_map['u']].reshape((1, -1))
        else:
            u = None
        if 'Time' in self.input_key_map.keys():
            Time = data[self.input_key_map['Time']]
        else:
            Time = None
        if 'd' in self.input_key_map.keys():
            d = data[self.input_key_map['d']].reshape((1, -1))
            out = self.model.simulate(nsim=1, U=u, D=d, x0=x0, Time=Time)
        elif u is not None:
            out = self.model.simulate(nsim=1, U=u, x0=x0, Time=Time)
        else:
            out = self.model.simulate(nsim=1, x0=x0, Time=Time)
        output = {}
        output[self.output_keys[0]] = out['X'][1, :]
        if out['Y'].shape[0] == 1:
            output[self.output_keys[1]] = out['Y'][0, :]
        else:
            output[self.output_keys[1]] = out['Y'][1, :]
        return output


class DynamicsNeuromancer(SystemDynamics):
    """
    Wrapper for Neuromancer state space model (SSM) classes defined in neuromancer.dynamics.py
    For complete documentaton of Neuromancer's SSM classes see:
    https://pnnl.github.io/neuromancer/dynamics.html
    """

    def __init__(self, model, input_key_map={}, name=None):
        """
        :param model: (callable) neuromancer Dynamics system model
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param extra_inputs: (list of str) Input keys to be added to canonical input.
        :param name: (str) Name for tracking output
        """
        super().__init__(input_key_map=input_key_map, name=name)
        self.model = model
        self.nx = model.nx
        self.ny = model.ny

    def _get_dims(self, data: Dict[str, np.ndarray]):
        nx = data[self.input_key_map['x']].shape[0]
        if 'u' in self.input_key_map.keys():
            nu = data[self.input_key_map['u']].shape[0]
        else:
            nu = 0
        if 'd' in self.input_key_map.keys():
            nd = data[self.input_key_map['d']].shape[0]
        else:
            nd = 0
        if 'w' in self.input_key_map.keys():
            nw = data[self.input_key_map['w']].shape[0]
        else:
            nw = 0
        return nx, nu, nd, nw

    def eval_model(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        nx, nu, nd, nw = self._get_dims(data)
        x0 = torch.tensor(data[self.input_key_map['x']].reshape((1, nx)))
        y = torch.ones(1, 1, self.ny)
        tensor_data = {self.model.input_key_map['x0']: x0.float(),
                       self.model.input_key_map['Yf']: y.float()}
        if 'u' in self.input_key_map.keys():
            u = torch.tensor(data[self.input_key_map['u']].reshape((1, 1, nu)))
            tensor_data[self.model.input_key_map['Uf']] = u.float()
        if 'd' in self.input_key_map.keys():
            d = torch.tensor(data[self.input_key_map['d']].reshape((1, 1, -1)))
            tensor_data[self.model.input_key_map['Df']] = d.float()
        if 'Time' in self.input_key_map.keys():
            Time = torch.tensor(data[self.input_key_map['Time']].reshape((1, 1, -1)))
            tensor_data[self.model.input_key_map['Time']] = Time.float()
        out = self.model(tensor_data)
        output = {}
        output[self.output_keys[0]] = out[self.model.output_keys[1]][0, 0, :].detach().numpy()
        output[self.output_keys[1]] = out[self.model.output_keys[2]][0, 0, :].detach().numpy()
        return output


class MovingHorizon(SystemComponent):
    DEFAULT_INPUT_KEYS = []
    OPTIONAL_INPUT_KEYS = []
    DEFAULT_OUTPUT_KEYS = []
    """
    Moving horizon class is creating sliding window of input time series data 
    with nstep horizon length.

    """
    def __init__(self, input_keys, output_keys=[], nsteps=1, name=None):
        """
        :param input_keys: (list (str)) input keys
        :param output_keys: (list (str)) output keys
        :param nsteps: (int) steps in the moving horizon window
        :param name: (str) Name for tracking output
        """
        super().__init__(name=name)
        self.nsteps = nsteps
        self.input_keys = input_keys
        if not output_keys:
            self.output_keys = [k+'_p' if name is None else k+'_'+name for k in input_keys]
        else:
            self.output_keys = output_keys
            assert len(self.output_keys) == len(self.input_keys), \
                f'Number of output keys {len(self.output_keys)} must equal to ' \
                f'number of input keys {len(self.input_keys)}.'
        self.mh_history = {key: [] for key in self.output_keys}

    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        output = {}
        for key_in, key_out in zip(self.input_keys, self.output_keys):
            self.mh_history[key_out] += [data[key_in]]
            if len(self.mh_history[key_out]) >= self.nsteps:
                output[key_out] = np.stack(self.mh_history[key_out][-self.nsteps:], axis=0)
            else:
                last_nsteps = np.stack(self.mh_history[key_out], axis=0)
                fill_void = np.repeat(last_nsteps[[0], :], self.nsteps - last_nsteps.shape[0], axis=0)
                output[key_out] = np.concatenate([fill_void, last_nsteps])
        return output


class SystemEstimator(SystemComponent):
    """
    SystemEstimator is a class that wraps state estimator callables.
    Estimators compute state variable estimates based on measurement data 
    given by variables defined in the input_keys.
    Estimator is defined as:
    x = estimator(m)
    where:
    estimator - state estimator callable
    x - estimated state variable
    m - measurement variables defined via input_keys
    """

    DEFAULT_INPUT_KEYS = []
    OPTIONAL_INPUT_KEYS = []
    DEFAULT_OUTPUT_KEYS = ["x"]

    def __init__(self, estimator, input_keys=[], name=None):
        """
        :param estimator: (callable) system estimator
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(name=name)
        self.input_keys = input_keys
        self.estimator = estimator

    def features(self, data):
        featlist = []
        for key in self.input_keys:
            assert isinstance(data[key], np.ndarray), \
                f'Estimator features must be np.ndarray type, ' \
                f'got {type(data[key])}'
            if len(data[key].shape) == 1:  # current time measurements
                featlist.append(data[key])
            elif len(data[key].shape) == 2:  # past sequence measurements
                featlist.append(data[key].flatten())
            else:
                raise ValueError(f"Estimator feature {key} has "
                                 f"{len(data[key].shape)} dimensions. "
                                 f"Should have 1 or 2 dimensions")
        return np.concatenate(featlist)

    @abstractmethod
    def eval_estimator(self, features: np.ndarray) -> np.ndarray:
        """
        implements forward pass estimator evaluation
        """
        pass

    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        output = {}
        features = self.features(data)
        x = self.eval_estimator(features)
        assert len(x.shape) == 1, \
            f'Estimated states need to have dimensions 1, got {x.shape}'
        output[self.output_keys[0]] = x
        return output


class EstimatorLinear(SystemEstimator):
    """
    Linear state estimator K.
        x = K*m
    where:
        K - state estimator gain
        x - estimated state variable
        m - measurement variables defined via input_keys
    """
    def __init__(self, estimator, input_keys=[], name=None):
        """
        :param estimator: (np.ndarray) linear matrix system estimator
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(estimator=estimator, input_keys=input_keys, name=name)
        assert isinstance(self.estimator, np.ndarray), \
            f'Linear estimator must be np.ndarray type, got {type(self.estimator)}'
        assert len(self.estimator.shape) == 2, \
            f'Linear estimator must be two dimensional np.ndarray, ' \
            f'got {len(self.estimator.shape)} dimensions'

    def eval_estimator(self, features: np.ndarray) -> np.ndarray:
        x = np.matmul(self.estimator, features)
        return x


class EstimatorCallable(SystemEstimator):
    """
    System estimator given as a generic python callable.
    """
    def __init__(self, estimator, input_keys=[], name=None):
        """
        :param estimator: (callable) system estimator
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(estimator=estimator, input_keys=input_keys, name=name)
        assert callable(self.estimator), \
            f'Estimator must be callable, got {type(self.estimator)}'

    def eval_estimator(self, features: np.ndarray) -> np.ndarray:
        x = self.estimator(features)
        return x


class EstimatorPytorch(SystemEstimator):
    """
    System estimator via generic Pytorch nn.Module
    """
    def __init__(self, estimator, input_keys=[], device="cpu", name=None):
        """
        :param estimator: (nn.Module) Pytorch system estimator
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        estimator = estimator.to(device)
        super().__init__(estimator=estimator, input_keys=input_keys, name=name)
        assert isinstance(self.estimator, nn.Module), \
            f'Estimator must be nn.Module, got {type(self.estimator)}'

    def eval_estimator(self, features: np.ndarray) -> np.ndarray:
        xi = torch.tensor(features).float()
        x = self.estimator(xi).detach().numpy()
        return x


class SystemController(SystemComponent):
    """
    Base control policy class mapping feature variables defined via input_keys
    onto control action variables u.
    Control policy is given as:
    u = policy(features)
    where:
    policy - control policy callable
    u - control action variables
    features - feature variables defined via input_keys
    """

    DEFAULT_INPUT_KEYS = []
    OPTIONAL_INPUT_KEYS = []
    DEFAULT_OUTPUT_KEYS = ["u"]

    def __init__(self, policy, input_keys=[], name=None):
        """
        :param policy: (callable) system policy
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(name=name)
        self.input_keys = input_keys
        self.policy = policy

    def rhc(self, u):
        assert 1 <= len(u.shape) <= 2, \
            f'Control policy actions need to have dimensions 1 or 2, ' \
            f'got {u.shape}'
        if len(u.shape) == 2:
            u = u[0, :]  # receding horizon control (RHC)
        return u

    def features(self, data):
        featlist = []
        for key in self.input_keys:
            assert isinstance(data[key], np.ndarray), \
                f'Control policy features must be np.ndarray type, ' \
                f'got {type(data[key])}'
            if len(data[key].shape) == 1:  # instantaneous measurements feedback
                featlist.append(data[key])
            elif len(data[key].shape) == 2:  # past sequence feedback
                featlist.append(data[key].flatten())
            else:
                raise ValueError(f"Control policy feature {key} has "
                                 f"{len(data[key].shape)} dimensions. "
                                 f"Should have 1 or 2 dimensions")
        return np.concatenate(featlist)

    @abstractmethod
    def eval_policy(self, features: np.ndarray) -> np.ndarray:
        """
        implements forward pass policy evaluation
        """
        pass

    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        output = {}
        features = self.features(data)
        u = self.eval_policy(features)
        output[self.output_keys[0]] = self.rhc(u)
        return output


class ControllerLinear(SystemController):
    """
    Linear control policy class:
        u = K*features
    where:
        K - control gain matrix
        u - control action variables
        features - feature variables defined via input_keys
    """
    def __init__(self, policy, input_keys=[], name=None):
        """
        :param policy: (np.array) linear system policy
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(policy=policy, input_keys=input_keys, name=name)
        assert isinstance(self.policy, np.ndarray), \
            f'Linear control policy must be np.ndarray type, got {type(self.policy)}'
        assert len(self.policy.shape) == 2, \
            f'Linear control policy must be two dimensional np.ndarray, ' \
            f'got {len(self.policy.shape)} dimensions'

    def eval_policy(self, features: np.ndarray) -> np.ndarray:
        u = np.matmul(self.policy, features)
        return u


class ControllerCallable(SystemController):
    """
    Control policy class via generic python callable
        u = policy(features)
    """
    def __init__(self, policy, input_keys=[], name=None):
        """
        :param policy: (callable) callable system policy
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        super().__init__(policy=policy, input_keys=input_keys, name=name)
        assert callable(self.policy), \
            f'Control policy must be callable, got {type(self.policy)}'

    def eval_policy(self, features: np.ndarray) -> np.ndarray:
        u = self.policy(features)
        return u


class ControllerPytorch(SystemController):
    """
    Control policy class via generic Pytorch nn.Module
        u = policy(features)
    """
    def __init__(self, policy, input_keys=[], nsteps=1, device="cpu", name=None):
        """
        :param policy: (nn.Module) Pytorch system policy
        :param input_keys: (list (str)) input keys as features to the estimator
        :param name: (str) Name for tracking output
        """
        policy = policy.to(device)
        super().__init__(policy=policy, input_keys=input_keys, name=name)
        assert isinstance(self.policy, nn.Module), \
            f'Control policy must be nn.Module, got {type(self.policy)}'
        self.nsteps = nsteps

    def eval_policy(self, features: np.ndarray) -> np.ndarray:
        xi = torch.tensor(features).float()
        u = self.policy(xi)
        u = u.reshape(self.nsteps, -1)
        return u.detach().numpy()


class SystemConstraints(SystemComponent):
    """
    Weapper class to evaluate Neuromancer constraints in the SystemSimulator.
    For a complete documentation of Neuromancer's constraints see:
    https://pnnl.github.io/neuromancer/constraint.html      
    """

    DEFAULT_INPUT_KEYS = []
    OPTIONAL_INPUT_KEYS = []
    DEFAULT_OUTPUT_KEYS = []

    def __init__(self, constraints, name=None):
        """
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        :param name: (str) Name for tracking output
        """
        super().__init__(name=name)
        self.constraints = constraints
        input_keys = []
        for con in self.constraints:
            input_keys += con.input_keys
        self.input_keys = list(set(input_keys))
        output_keys = []
        for con in self.constraints:
            output_keys += con.output_keys
        self.output_keys = list(set(output_keys))

    def eval_constraints(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        implements forward pass policy evaluation
        """
        output_dict = {}
        for c in self.constraints:
            output = c(input_dict)
            output_dict = {**output_dict, **output}
        return output_dict

    def _forward_step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_dict = self.pre_process(data)
        output_dict = self.eval_constraints(input_dict)
        output = self.post_process(output_dict)
        return output

    def pre_process(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        data = {k: torch.tensor(v).float() for k, v in data.items()}
        return data

    def post_process(self, data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        data = {k: v.detach().numpy() for k, v in data.items()}
        return data


if __name__ == "__main__":
    from neuromancer import dynamics, estimators, integrators, blocks
    from neuromancer.psl import systems

    """
    Test SystemLinSSM class
    """
    print('\ntest SystemLinSSM class \n')
    A = np.array([[0.8, 0.1],
                  [0.0, 0.8]])
    B = np.array([[0.0],
                  [0.5]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    ssm_system = DynamicsLinSSM(A, B, C)
    data = {'x': np.ones(ssm_system.nx),
            'u': np.ones(ssm_system.nu)}
    for k in range(30):
        out = ssm_system.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test SystemPSL class
    """
    print('\ntest SystemPSL class \n')
    psl_model = systems['LorenzControl']()
    psl_system = DynamicsPSL(psl_model, input_key_map={'u': 'u'})
    data = {'x': psl_model.x0,
            'u': psl_model.U[0]}
    for k in range(30):
        out = psl_system.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test SystemNeuromancer class with State Space Models
    """
    print('\ntest SystemNeuromancer class with State Space Models \n')
    nx, nu, ny, nd = 3, 2, 2, 1
    hsize, nlayers = 10, 2
    hsizes = [hsize for k in range(nlayers)]
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=hsizes) for insize in [nx, nu, nd]]
    fy = blocks.MLP(nx, ny, hsizes=hsizes)
    model = dynamics.BlockSSM(fx, fy, fu=fu, fd=fd, name='block_ssm')
    nm_system = DynamicsNeuromancer(model, input_key_map={'u': 'u', 'd': 'd'})
    data = {'x': np.ones(nx), 'u': np.ones(nu), 'd': np.ones(nd)}
    for k in range(30):
        out = nm_system.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test ControllerLinear class
    """
    print('\ntest ControllerLinear class \n')
    nx, nu, nd = 3, 2, 1
    K = np.random.rand(nu, nx + nd)
    policy_lin = ControllerLinear(policy=K, input_keys=['x', 'd'])
    data = {'x': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = policy_lin.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        print(data['x'])
        print(data['u'])

    """
    Test ControllerCallable class
    """
    print('\ntest ControllerCallable class \n')
    nx, nd = 3, 1
    func = lambda x: x - x ** 2
    policy_call = ControllerCallable(policy=func, input_keys=['x', 'd'])
    data = {'x': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = policy_call.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        print(data['x'])
        print(data['u'])

    """
    Test ControllerPytorch class with nm.blocks
    """
    print('\ntest ControllerPytorch class with nm.blocks \n')
    nx, nd, nu, n_hidden = 3, 1, 2, 10
    block = blocks.MLP(nx + nd, nu, hsizes=[n_hidden, n_hidden])
    policy_torch = ControllerPytorch(policy=block, input_keys=['x', 'd'])
    data = {'x': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = policy_torch.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        print(data['x'])
        print(data['u'])

    """
    Test ControllerPytorch class with nn.Module
    """
    print('\ntest ControllerPytorch class with nn.Module \n')
    nx, nd, nu, n_hidden = 3, 1, 2, 10
    controller = nn.Sequential(nn.Linear(nx + nd, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, nu),
                               nn.Sigmoid())
    policy_torch2 = ControllerPytorch(policy=controller, input_keys=['x', 'd'])
    data = {'x': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = policy_torch2.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        print(data['x'])
        print(data['u'])

    """
    Test EstimatorLinear class
    """
    print('\ntest EstimatorLinear class \n')
    nx, nd = 3, 1
    K = np.random.rand(nx, nx + nd)
    estim_lin = EstimatorLinear(estimator=K, input_keys=['y', 'd'])
    data = {'y': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = estim_lin.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test EstimatorCallable class
    """
    print('\ntest EstimatorCallable class \n')
    nx, nd = 3, 1
    func = lambda x: x - x ** 2
    estim_call = EstimatorCallable(estimator=func, input_keys=['y', 'd'])
    data = {'y': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = estim_call.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test EstimatorPytorch class with nm.blocks
    """
    print('\ntest EstimatorPytorch class with nm.blocks \n')
    nx, nd, n_hidden = 3, 1, 10
    block = blocks.MLP(nx + nd, nu, hsizes=[n_hidden, n_hidden])
    estim_torch = EstimatorPytorch(estimator=block, input_keys=['y', 'd'])
    data = {'y': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = estim_torch.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test EstimatorPytorch class with nn.Module
    """
    print('\ntest EstimatorPytorch class with nn.Module \n')
    nx, nd, nu, n_hidden = 3, 1, 2, 10
    estimator = nn.Sequential(nn.Linear(nx + nd, n_hidden),
                              nn.ReLU(),
                              nn.Linear(n_hidden, nu),
                              nn.Sigmoid())
    estim_torch2 = EstimatorPytorch(estimator=estimator, input_keys=['y', 'd'])
    data = {'y': np.ones(nx), 'd': np.ones(nd)}
    for k in range(30):
        out = estim_torch2.step(data)
        data = {**data, **out}
        print(data['x'])

    """
    Test Constraints class with neuromancer constraints
    """
    from neuromancer.constraint import variable

    nx = 3
    x = variable("x")
    xmin = variable("xmin")
    xmax = variable("xmax")
    con1 = (x > xmin)
    con2 = (x < xmax)
    constraints = [con1, con2]
    system_con = SystemConstraints(constraints)
    data = {'x': np.random.randn(nx),
            'xmin': -np.ones(nx), 'xmax': np.ones(nx)}
    for k in range(30):
        out = system_con.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        for key in constraints[0].output_keys:
            print(f'{key}: {data[key]}')

    """
    Test Constraints class with neuromancer objectives
    """
    nx = 3
    x = variable("x")
    y = variable("y")
    f1 = (1 - x) ** 2 + (y - x ** 2) ** 2
    obj1 = f1.minimize(weight=1., name='obj1')
    f2 = x ** 2 + y ** 2
    obj2 = f2.minimize(weight=1., name='obj2')
    objectives = [obj1, obj2]
    system_obj = SystemConstraints(objectives)
    data = {'x': np.random.randn(nx), 'y': np.random.randn(nx)}
    for k in range(30):
        out = system_obj.step(data)
        data = {**data, **out}
        data['x'] += np.random.randn(nx)
        for key in objectives[0].output_keys:
            print(f'{key}: {data[key]}')

    """
    Test loop ControllerLinear class + SystemLinSSM class
    """
    print('\ntest ControllerLinear class + SystemLinSSM class \n')
    A = np.array([[0.6, 0.1],
                  [0.0, 0.8]])
    B = np.array([[0.0],
                  [0.5]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    ssm_system = DynamicsLinSSM(A, B, C)
    nx, nu = A.shape[0], B.shape[1]
    K = np.random.rand(nu, nx)
    policy_lin = ControllerLinear(policy=K, input_keys=['x'])
    components = [policy_lin, ssm_system]
    input_dict = {'x': np.ones(ssm_system.nx)}
    steps = [input_dict]
    for k in range(30):
        for comp in components:
            output_dict = comp.step(input_dict)
            input_dict = {**input_dict, **output_dict}
        print(input_dict['x'])
        print(input_dict['u'])
        steps.append(input_dict)

    """
    Test SystemSimulator class: ControllerLinear + SystemLinSSM
    """
    print('\ntest SystemSimulator class: '
          'ControllerLinear + SystemLinSSM \n')
    cl_sim = SystemSimulator(components)
    data = {'x': np.ones(ssm_system.nx)}
    # test step method
    for k in range(30):
        data = cl_sim.step(data)
        print(data['x'])
        print(data['u'])
    # test simulate method
    data_init = {'x': np.ones(ssm_system.nx)}
    trajectories = cl_sim.simulate(nsim=20, data_init=data_init)
    print(trajectories['y'].shape)
    print(trajectories['x'].shape)
    print(trajectories['u'].shape)

    """
    Test SystemSimulator class: SystemPSL + SystemNeuromancer for Autonomous
    """
    print('\ntest SystemSimulator class: '
          'SystemPSL + SystemNeuromancer (Autonomous) \n')
    psl_model = systems['Brusselator1D']()
    psl_system = DynamicsPSL(psl_model, name='psl', input_key_map={'x': 'x_psl'})
    nx = psl_model.nx
    hsize, nlayers = 10, 2
    hsizes = [hsize for k in range(nlayers)]
    fx = blocks.MLP(nx, nx, hsizes=hsizes)
    fy = blocks.MLP(nx, nx, hsizes=hsizes)
    model = dynamics.BlockSSM(fx, fy, name='block_ssm')
    nm_system = DynamicsNeuromancer(model, name='nm', input_key_map={'x': 'x_nm'})
    components = [nm_system, psl_system]
    ol_system = SystemSimulator(components)
    x0 = np.asarray(psl_model.x0)
    data_init = {'x_psl': x0, 'x_nm': x0}
    trajectories = ol_system.simulate(nsim=200, data_init=data_init)

    """
    Test SystemSimulator class: SystemPSL + SystemNeuromancer for NonAutonomous
    """
    print('\ntest SystemSimulator class: '
          'SystemPSL + SystemNeuromancer (NonAutonomous) \n')
    import neuromancer.slim as slim
    from neuromancer import ode

    # PSL model
    system = systems['TwoTank']
    ts = 1.0
    nsim = 2000
    psl_model = system(ts=ts, nsim=nsim)
    psl_system = DynamicsPSL(psl_model, name='psl',
                             input_key_map={'x': 'x_psl', 'u': 'u'})
    nx = psl_model.nx
    nu = psl_model.nu
    # neuromancer ODE model
    interp_u = lambda tq, t, u: u
    two_tank_ode = ode.TwoTankParam()
    two_tank_ode.c1 = nn.Parameter(torch.tensor([psl_model.c1]), requires_grad=False)
    two_tank_ode.c2 = nn.Parameter(torch.tensor([psl_model.c2]), requires_grad=False)
    fx_int = integrators.RK4(two_tank_ode, interp_u=interp_u, h=psl_model.ts)
    fy = slim.maps['identity'](nx, nx)
    dyn_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                                    input_key_map={"x0": "x1", 'Uf': 'Uf1'},
                                    name='dynamics', online_flag=False)
    nm_system = DynamicsNeuromancer(dyn_model, name='nm',
                                    input_key_map={'x': 'x_nm', 'u': 'u'})
    # CL simulator
    components = [nm_system, psl_system]
    ol_system = SystemSimulator(components)
    # simulate
    sim_steps = 1000
    x0 = np.asarray(psl_model.x0)
    data_init = {'x_psl': x0, 'x_nm': x0}
    raw = psl_model.simulate()
    data_traj = {'u': raw['U'][:sim_steps + 1, :]}
    trajectories = ol_system.simulate(nsim=sim_steps, data_init=data_init,
                                      data_traj=data_traj)

    """
    Test MovingHorizon
    """
    print('\ntest MovingHorizon class \n')
    # psl system
    system = systems['TwoTank']
    ts = 1.0
    nsim = 2000
    psl_model = system(ts=ts, nsim=nsim)
    psl_system = DynamicsPSL(psl_model, name='psl',
                             input_key_map={'x': 'x_psl', 'u': 'u'})
    # moving horizon
    nsteps = 5
    mh = MovingHorizon(input_keys=['x_psl'], nsteps=nsteps)
    # neuromancer estimator
    nx, n_hidden = psl_model.nx, 10
    estim = estimators.MLPEstimator({"x0": (nx,), 'xp': (1, nx)},
        nsteps=nsteps, window_size=nsteps, input_keys=["xp"], name='estim')
    estim_nm = EstimatorPytorch(estimator=estim.net,
                                input_keys=['x_psl_p'], name='estim')
    # block estimator
    block = blocks.MLP(nsteps*nx, nx, hsizes=[n_hidden, n_hidden])
    estim_nm2 = EstimatorPytorch(estimator=block, input_keys=['x_psl_p'], name='estim2')
    # neuromancer ODE model
    two_tank_ode = ode.TwoTankParam()
    two_tank_ode.c1 = nn.Parameter(torch.tensor([psl_model.c1]), requires_grad=False)
    two_tank_ode.c2 = nn.Parameter(torch.tensor([psl_model.c2]), requires_grad=False)
    fx_int = integrators.RK4(two_tank_ode, interp_u=lambda tq, t, u: u, h=psl_model.ts)
    fy = slim.maps['identity'](nx, nx)
    dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                                         input_key_map={"x0": 'x0',
                                                        'Yf': 'Yf', 'Uf': 'Uf'},
                                         name='dynamics', online_flag=False)
    nm_system = DynamicsNeuromancer(dyn_model, name='nm',
                                    input_key_map={'x': 'x_estim', 'u': 'u'})
    components = [mh, estim_nm, estim_nm2, nm_system, psl_system]
    system_sim = SystemSimulator(components)
    sim_steps = 1800
    x0 = np.asarray(psl_model.x0)
    data_init = {'x_psl': x0, 'x_nm': x0}
    U = raw['U'][:sim_steps + 1, :]
    data_traj = {'u': U}
    trajectories = system_sim.simulate(nsim=sim_steps, data_init=data_init,
                                       data_traj=data_traj)

