
import torch
import torch.nn as nn
from neuromancer.gradients import gradient, jacobian
from abc import ABC, abstractmethod
from functorch import jacrev, jacfwd


class Solver(nn.Module, ABC):
    """
    Abstract class for the differentiable solver implementation
    """
    def __init__(self, objectives=[], constraints=[],
                 input_keys=[], output_keys=[], name=None):
        """

        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        :param input_keys: (list of str) For gathering inputs from intermediary data dictionary
        :param output_keys: (list of str) For sending inputs to other nodes through intermediary data dictionary
        :param name: (str) Unique identifier
        """
        super().__init__()
        # input_keys are keys of input variables to be updated by the solver
        self.input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        # output_keys are keys associated with updated values of the input variables
        self.output_keys = output_keys if output_keys else input_keys
        assert len(self.input_keys) == len(self.output_keys), \
            f'Length of input_keys {self.input_keys} must equal length of output_keys {self.output_keys}'
        self.name = name
        self.objectives = nn.ModuleList(objectives)     # list of neuromancer objectives
        self.constraints = nn.ModuleList(constraints)   # list of neuromancer constraints

    @abstractmethod
    def forward(self, data):
        """
        differentiable solver update to be implemented here

        :param datadict: (dict {str: Tensor}) input to solver with associated input_keys
        :return: (dict {str: Tensor}) Output of solver with associated output_keys
        """
        pass


class GradientProjection(Solver):
    """
    Implementation of projected gradient method for gradient-based corrections of constraints violations
    Abstract steps of the gradient projection method:
        1, compute aggregated constraints violation penalties (con_viol_energy method)
        2, compute gradient of the constraints violations w.r.t. variables in input_keys (forward method)
        3, update the variable values with the negative gradient scaled by step_size (forward method)

    References:
        method: https://neos-guide.org/guide/algorithms/gradient-projection/
        DC3 paper: https://arxiv.org/abs/2104.12225
    """
    def __init__(self, constraints, input_keys, output_keys=[], decay=0.1,
                 num_steps=1, step_size=0.01, energy_update=True, name=None):
        """
        :param constraints:
        :param input_keys: (List of str) List of input variable names
        :param constraints: list of objects which implement the Loss interface (e.g. Objective, Loss, or Constraint)
        :param num_steps: (int) number of iteration steps for the projected gradient method
        :param step_size: (float) scaling factor for gradient update
        :param decay: (float) decay factor of the step_size
        :param energy_update: (bool) flag to update energy
        :param name:
        """
        super().__init__(constraints=constraints,
                         input_keys=input_keys, output_keys=output_keys,
                         name=name)
        self.num_steps = num_steps
        self.step_size = step_size
        self.input_keys = input_keys
        self.decay = decay
        self.energy_update = energy_update

    def _constraints_check(self):
        """
        :return:
        """
        for con in self.constraints:
            assert str(con.comparator) in ['lt', 'gt'], \
                f'constraint {con} must be inequality (lt or gt), but it is {str(con.comparator)}'

    def con_viol_energy(self, input_dict):
        """
        Calculate the constraints violation potential energy over batches
        """
        C_violations = []
        for con in self.constraints:
            output = con(input_dict)
            cviolation = output[con.output_keys[2]]
            C_violations.append(cviolation.reshape(cviolation.shape[0], -1))
        C_violations = torch.cat(C_violations, dim=-1)
        energy = torch.mean(torch.abs(C_violations), dim=1)
        return energy

    def forward(self, data):
        """
        forward pass of the projected gradient solver
        :param data: (dict: {str: Tensor})
        :return: (dict: {str: Tensor})
        """
        # init output
        output_data = data.copy()
        if self.energy_update:
            data = output_data
        # init decay rate
        d = 1
        # projected gradient
        for k in range(self.num_steps):
            # update energy
            energy = self.con_viol_energy(data)
            for in_key, out_key in zip(self.input_keys, self.output_keys):
                # get grad
                x = data[in_key]
                step = gradient(energy, x)
                assert step.shape == x.shape, \
                    f'Dimensions of gradient step {step.shape} should be equal to dimensions ' \
                    f'{x.shape}  of a single variable {in_key}'
                # update
                x = x - d * self.step_size*step
                d = d - self.decay * d
                output_data[out_key] = x
        return output_data


class IterativeSolver(nn.Module):
    """
    TODO: to debug

    Class for a family of iterative solvers for root-finding solutions to the problem:
        :math:`g(x) = 0`

    general iterative solver update rules:
    :math:`x_k+1 = phi(x_k)`
    :math:`x_k+1 = x_k + phi(x_k)`

    https://en.wikipedia.org/wiki/Iterative_method
    https://en.wikipedia.org/wiki/Root-finding_algorithms

    Newton's method:
    :math:`x_k+1 = x_k - J_g(x_k)^-1 g(x_k)`
    :math:`J_g(x_k)`: Jacobian of :math:`g(x_k)` w.r.t. :math:`x_k'


    """
    def __init__(self, constraints, input_keys, output_keys=[],
                 num_steps=1, step_size=1., name=None):
        """

        :param constraints:
        :param input_keys: (List of str) List of input variable names
        :param constraints: list of objects which implement the Loss interface (e.g. Objective, Loss, or Constraint)
        :param num_steps: (int) number of iteration steps for the projected gradient method
        :param step_size: (float) scaling factor for gradient update
        :param name:
        """
        super().__init__()
        self.input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        self.output_keys = output_keys if output_keys else input_keys
        assert len(self.input_keys) == len(self.output_keys), \
            f'Length of input_keys {self.input_keys} must equal length of output_keys {self.output_keys}'
        self.name = name
        self.constraints = nn.ModuleList(constraints)
        self._num_steps = num_steps
        self.input_keys = input_keys
        self.step_size = step_size
        self._constraints_check()

    def _constraints_check(self):
        """
        :return:
        """
        for con in self.constraints:
            assert str(con.comparator) == 'eq', \
                f'constraint {con} must be equality (eq), but it is {str(con.comparator)}'

    @property
    def num_steps(self):
        return self._num_steps

    def con_values(self, data):
        """
        Calculate values g(x) of the constraints expressions
        """
        g_values = []
        for con in self.constraints:
            output = con(data)
            cvalue = output[con.output_keys[1]]
            g_values.append(cvalue.reshape(cvalue.shape[0], -1))
        return torch.cat(g_values, dim=-1)

    # TODO: debug this with functorch
    def newton_step(self, data, x):
        """
        Calculate the newton step for a given variable x
        """
        g = self.con_values(data)
        # J_g = jacobian(g, x)
        J_g = jacfwd(self.con_values, argnums=0)(x)
        step = torch.matmul(J_g.inverse(), g)
        return step

    def forward(self, data):
        """
        foward pass of the Newton solver
        :param data: (dict: {str: Tensor})
        :return: (dict: {str: Tensor})
        """
        output_data = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            x = data[in_key]
            for k in range(self.num_steps):
                step = self.newton_step(data, x)
                assert step.shape == x.shape, \
                    f'Dimensions of solver step {step.shape} should be equal to dimensions ' \
                    f'{x.shape} of a single variable {in_key}'
                x = x - self.step_size*step
            output_data[out_key] = x
        return output_data

