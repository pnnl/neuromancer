
import torch
import torch.nn as nn
from neuromancer.gradients import gradient, jacobian
from neuromancer.component import Component


class GradientProjection(Component):
    """
    Implementation of projected gradient method for gradient-based corrections of constraints violations
    Abstract steps of the gradient projection method:
    1, compute aggregated constraints violation penalties
    2, compute gradient of the constraints violations w.r.t. variables in input_keys
    3, update the variable values with the negative gradient scaled by step_size
    method reference: https://neos-guide.org/content/gradient-projection-methods
    used in DC3 paper: https://arxiv.org/abs/2104.12225
    """
    def __init__(self, constraints, input_keys, output_keys=[], decay=0.1,
                 num_steps=1, step_size=0.01, batch_second=False, name=None):
        """
        :param constraints:
        :param input_keys: (List of str) List of input variable names
        :param constraints: list of objects which implement the Loss interface (e.g. Objective, Loss, or Constraint)
        :param num_steps: (int) number of iteration steps for the projected gradient method
        :param step_size: (float) scaling factor for gradient update
        :param decay:
        :param batch_second:
        :param name:
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        output_keys = output_keys if output_keys else input_keys
        assert len(input_keys) == len(output_keys), \
            f'Length of input_keys {input_keys} must equal length of output_keys {output_keys}'

        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.constraints = nn.ModuleList(constraints)
        self._num_steps = num_steps
        self.step_size = step_size
        self.input_keys = input_keys
        self.batch_second = batch_second
        self.decay = decay
        # self._constraints_check()

    def _constraints_check(self):
        """
        :return:
        """
        for con in self.constraints:
            assert str(con.comparator) in ['lt', 'gt'], \
                f'constraint {con} must be inequality (lt or gt), but it is {str(con.comparator)}'

    @property
    def num_steps(self):
        return self._num_steps

    def con_viol_energy(self, input_dict):
        """
        Calculate the constraints violation potential energy over batches
        """
        C_violations = []
        for con in self.constraints:
            output = con(input_dict)
            cviolation = output[con.output_keys[2]]
            if self.batch_second:
                cviolation = cviolation.transpose(0, 1)
            C_violations.append(cviolation.reshape(cviolation.shape[0], -1))
        C_violations = torch.cat(C_violations, dim=-1)
        energy = torch.mean(torch.abs(C_violations), dim=1)
        return energy

    def forward(self, input_dict):
        energy = self.con_viol_energy(input_dict)
        output_dict = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            x = input_dict[in_key]
            step = gradient(energy, x)
            assert step.shape == x.shape, \
                f'Dimensions of gradient step {step.shape} should be equal to dimensions ' \
                f'{x.shape}  of a single variable {in_key}'
            d = 1.
            for k in range(self.num_steps):
                x = x - d*self.step_size*step
                d = d - self.decay*d
            output_dict[out_key] = x
        return output_dict


class IterativeSolver(Component):
    """
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
                 num_steps=1, step_size=1., batch_second=False, name=None):
        """

        :param constraints:
        :param input_keys: (List of str) List of input variable names
        :param constraints: list of objects which implement the Loss interface (e.g. Objective, Loss, or Constraint)
        :param num_steps: (int) number of iteration steps for the projected gradient method
        :param step_size: (float) scaling factor for gradient update
        :param batch_second:
        :param name:
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        output_keys = output_keys if output_keys else input_keys
        assert len(input_keys) == len(output_keys), \
            f'Length of input_keys {input_keys} must equal length of output_keys {output_keys}'
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.constraints = nn.ModuleList(constraints)
        self._num_steps = num_steps
        self.input_keys = input_keys
        self.step_size = step_size
        self.batch_second = batch_second
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

    def newton_step(self, input_dict, x):
        """
        Calculate the newton step for a given variable x
        """
        g_values = []
        for con in self.constraints:
            output = con(input_dict)
            cvalue = output[con.output_keys[1]]
            if self.batch_second:
                cvalue = cvalue.transpose(0, 1)
            g_values.append(cvalue.reshape(cvalue.shape[0], -1))
        g = torch.cat(g_values, dim=-1)
        J_g = jacobian(g, x)
        step = torch.matmul(J_g.inverse(), g)
        return step

    def forward(self, input_dict):
        output_dict = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            x = input_dict[in_key]
            for k in range(self.num_steps):
                step = self.newton_step(input_dict, x)
                assert step.shape == x.shape, \
                    f'Dimensions of solver step {step.shape} should be equal to dimensions ' \
                    f'{x.shape} of a single variable {in_key}'
                x = x - self.step_size*step
            output_dict[out_key] = x
        return output_dict

