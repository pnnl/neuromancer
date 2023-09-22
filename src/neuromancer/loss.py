"""
Loss function aggregators that create physics-informed loss functions
from the list of defined objective terms and constraints.

Currently supported loss functions:

* `PenaltyLoss <https://en.wikipedia.org/wiki/Penalty_method>`_
* `BarrierLoss <https://en.wikipedia.org/wiki/Barrier_function>`_
* `AugmentedLagrangeLoss <https://en.wikipedia.org/wiki/Augmented_Lagrangian_method>`_

"""

import math
from abc import ABC, abstractmethod
import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from neuromancer.constraint import Constraint


class AggregateLoss(nn.Module, ABC):
    """
    Abstract aggregate loss class for calculating constraints, objectives, and aggegate loss values.
    """
    def __init__(self, objectives, constraints):
        """
        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        """
        super().__init__()
        self.objectives = nn.ModuleList(objectives)
        for c in constraints:
            assert isinstance(c, Constraint)
        self.constraints = nn.ModuleList(constraints)

        input_keys = []
        for obj in self.objectives:
            input_keys += obj.input_keys
        for con in self.constraints:
            input_keys += con.input_keys
        self.input_keys = list(set(input_keys))
        self.output_keys = ['loss', 'objective_loss', 'penalty_loss',
                            'C_violations', 'C_values', 'C_eq_violations',
                            'C_ineq_violations', 'C_eq_values', 'C_ineq_values']
        self._check_keys()

    def _check_keys(self):
        keys = set()
        for loss in list(self.objectives) + list(self.constraints):
            keys |= set(loss.input_keys)
            new_keys = set(loss.output_keys)
            same = new_keys & keys
            assert len(same) == 0, \
                f'Keys {same} are being overwritten by the loss term {loss}.'
            keys |= new_keys

    def calculate_objectives(self, input_dict):
        """
        Calculate the value of the objective function for SGD
        """
        loss = 0.0
        output_dict = {}
        for objective in self.objectives:
            output = objective(input_dict)
            if isinstance(output, torch.Tensor):
                output = {objective.output_keys[0]: output}
            output_dict = {**output_dict, **output}
            loss += output_dict[objective.output_keys[0]]
        output_dict['objective_loss'] = loss
        return output_dict

    def calculate_constraints(self, input_dict):
        """
        Calculate the values of constraints and constraints violations
        """
        loss = 0.0
        output_dict = {}
        C_values = []
        C_violations = []
        eq_flags = []
        for c in self.constraints:
            # get loss, values, and violations of constraint via its forward pass
            output = c(input_dict)
            output_dict = {**output_dict, **output}
            loss += output[c.output_keys[0]]
            cvalue = output[c.output_keys[1]]
            cviolation = output[c.output_keys[2]]
            nr_constr = math.prod(cvalue.shape[1:])
            eq_flags += nr_constr*[str(c.comparator) == 'eq']
            C_values.append(cvalue.reshape(cvalue.shape[0], -1))
            C_violations.append(cviolation.reshape(cviolation.shape[0], -1))
        if self.constraints:
            equalities_flags = np.array(eq_flags)
            # get aggregated constraints
            C_violations = torch.cat(C_violations, dim=-1)
            C_values = torch.cat(C_values, dim=-1)
            output_dict['C_violations'] = C_violations
            output_dict['C_values'] = C_values
            output_dict['C_eq_violations'] = C_violations[:, equalities_flags]
            output_dict['C_ineq_violations'] = C_violations[:, ~equalities_flags]
            output_dict['C_eq_values'] = C_values[:, equalities_flags]
            output_dict['C_ineq_values'] = C_values[:, ~equalities_flags]
        output_dict['penalty_loss'] = loss
        return output_dict

    @abstractmethod
    def forward(self, input_dict):
        pass

    def __add__(self, other):
        """
        Overload the + operator to aggregate objective functions and constraints
        """
        if not isinstance(other, type(self)):
            raise ValueError("Only instances of the same loss can be added")
        # unique names
        obj1 = nn.ModuleList([copy.copy(obj) for obj in self.objectives])
        for obj in obj1:
            obj.name += "_{}".format(id(self))
        obj2 = nn.ModuleList([copy.copy(obj) for obj in other.objectives])
        for obj in obj2:
            obj.name += "_{}".format(id(other))
        cons1 =  nn.ModuleList([copy.copy(con) for con in self.constraints])
        for con in cons1:
            con.name += "_{}".format(id(self))
        cons2 = nn.ModuleList([copy.copy(con) for con in other.constraints])
        for con in cons2:
            con.name += "_{}".format(id(other))
        # combine objectives and constraints from both instances
        new_objectives = obj1 + obj2
        new_constraints = cons1 + cons2
        return type(self)(new_objectives, new_constraints)

    def __mul__(self, weight):
        """
        Overload the * operator to change the scale of objective functions and constraints
        """
        new_objectives = nn.ModuleList([weight * obj for obj in self.objectives])
        new_constraints = nn.ModuleList([weight * con for con in self.constraints])
        return type(self)(new_objectives, new_constraints)

    def __rmul__(self, weight):
        """
        Overload the * operator to change the scale of objective functions and constraints
        """
        new_objectives = nn.ModuleList([weight * obj for obj in self.objectives])
        new_constraints = nn.ModuleList([weight * con for con in self.constraints])
        return type(self)(new_objectives, new_constraints)


class PenaltyLoss(AggregateLoss):
    """
    Penalty loss function.
        https://en.wikipedia.org/wiki/Penalty_method
    """

    def __init__(self, objectives, constraints):
        """
        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        """
        super().__init__(objectives, constraints)

    def forward(self, input_dict):
        """

        :param input_dict: (dict {str: torch.Tensor}) Values from forward pass calculations
        :return: (dict {str: torch.Tensor}) input_dict appended with calculated loss values
        """
        objectives_dict = self.calculate_objectives(input_dict)
        input_dict = {**input_dict, **objectives_dict}
        fx = objectives_dict['objective_loss']
        penalties_dict = self.calculate_constraints(input_dict)
        input_dict = {**input_dict, **penalties_dict}
        penalties = penalties_dict['penalty_loss']
        input_dict['loss'] = fx + penalties
        return input_dict


class BarrierLoss(PenaltyLoss):
    """
    Barrier loss function.
    * https://en.wikipedia.org/wiki/Barrier_function
    Available barrier functions are defined in the self.barriers dictionary.
    References for relaxed barrier functions:
    * https://arxiv.org/abs/1602.01321
    * https://arxiv.org/abs/1904.04205v2
    * https://ieeexplore.ieee.org/document/7493643/
    """

    def __init__(self, objectives, constraints, barrier='log10', upper_bound=1.,
                 shift=1., alpha=0.5):
        """
        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        :param barrier: (string) type of the barrier function
        :param upper_bound (scalar) upper bound for the barrier function value
        :param shift (scalar) shift of the expshift barrier function towards the left
        :param alpha (scalar) bending of the soft exponential function
        """
        super().__init__(objectives, constraints)

        # choices of barrier functions
        #   warning: log10, log, inverse, and softlog might get numerically unstable
        #   softexp is numerically stable and thus a prefered option
        self.shift = shift
        self.alpha = alpha
        self.barriers = {'log10': lambda value: -torch.log10(-value),
                    'log': lambda value: -torch.log(-value),
                    'inverse': lambda value: 1 / (-value),
                    'softexp': lambda value: (torch.exp(self.alpha * value) - 1) / self.alpha + self.alpha,
                    'softlog': lambda value: -torch.log(1 + self.alpha * (-value - self.alpha)) / self.alpha,
                    'expshift': lambda value: torch.exp(value + self.shift)
                    }
        self.barrier = self._set_barrier(barrier)
        self.upper_bound = upper_bound

    def _set_barrier(self, barrier):
        if barrier in self.barriers:
            return self.barriers[barrier]
        else:
            assert callable(barrier), \
                f'The barrier, {barrier} must be a key in {self.barriers} or a callable.'
            return barrier

    def calculate_constraints(self, input_dict):
        """
        Calculate the magnitudes of constraint violations via log barriers
            cviolation > 0 -> penalty
            cviolation <= 0 -> barrier
        """
        loss = 0.0
        b_loss = 0.0
        output_dict = super().calculate_constraints(input_dict)
        for c in self.constraints:
            cvalue = output_dict[c.output_keys[1]]
            cviolation = output_dict[c.output_keys[2]]
            penalty_mask = cvalue >= 0
            cbarrier = self.barrier(cvalue)
            cbarrier[cbarrier != cbarrier] = 0.0  # replacing nan with 0 -> infeasibility
            cbarrier[cbarrier == float("Inf")] = 0.0  # replacing inf with 0 -> active constraints
            cbarrier = torch.clamp(cbarrier, min=0.0, max=self.upper_bound)
            output_dict[f'{c.name}_barrier'] = cbarrier
            if penalty_mask.any():
                penalty_loss = c.weight * torch.mean(penalty_mask * cviolation)
                loss += penalty_loss
            if (~penalty_mask).any():
                barrier_loss = c.weight * torch.mean(~penalty_mask * cbarrier)
                b_loss += barrier_loss
                loss += barrier_loss
        output_dict['barrier_loss'] = b_loss
        output_dict['penalty_loss'] = loss
        return output_dict


class AugmentedLagrangeLoss(AggregateLoss):
    """
    Augmented Lagrangian method loss function.
        https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
    """

    def __init__(self, objectives, constraints, train_data, inner_loop=10, sigma=2.,
                 mu_max=1000., mu_init=0.001, eta=1.0):
        """
        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        :param train_data: (torch DataLoader)
        :param inner_loop: (int) Number of iterations for the inner loop optimization. Lagrange multipliers
                                 are updated in the outer loop every inner_loop iterations.
        :param sigma: (float) Scaling factor for adaptive mu value. Shoud be > 1.0
        :param mu_max: (float) Maximum weight on constraint violations for optimization
        :param mu_init: (float) Initial weight on constraint violations for optimization
        :param eta: (float) Expected proportion of reduction in constraints violation. Should be <= 1.
        """
        super().__init__(objectives, constraints)
        self.inner_loop = inner_loop
        self.init = True
        self.register_buffer('nsamples', torch.tensor(len(train_data.dataset)))
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('mu_max', torch.tensor(mu_max))
        self.register_buffer('mu', mu_init * torch.ones(self.nsamples, 1))
        self.register_buffer('lm', torch.tensor(torch.empty(0)))
        self.register_buffer('eta', torch.tensor(eta))
        self.register_buffer('penalty_best', torch.tensor(np.finfo(np.float32).max))
        self.output_keys += ['2norm_unscaled_penalty_loss', 'unscaled_penalty_loss', 'scaled_performance',
                             'con_lagrangian', 'mu_scaled_penalty_loss', 'mu']
        self._check_keys()

    def forward(self, input_dict):
        objectives_dict = self.calculate_objectives(input_dict)
        input_dict = {**input_dict, **objectives_dict}
        fx = objectives_dict['objective_loss']
        input_dict['loss'] = fx

        con_dict = self.calculate_constraints(input_dict)
        input_dict = {**input_dict, **con_dict}
        C = con_dict['C_values']
        C_violations = con_dict['C_violations']
        scaled_penalty_loss = con_dict['penalty_loss']
        penalties = torch.sum(C_violations ** 2, dim=-1, keepdim=True)
        penalties_sqrt = torch.sqrt(penalties)
        unscaled_penalty_loss = torch.mean(penalties)
        input_dict['2norm_unscaled_penalty_loss'] = torch.mean(penalties_sqrt)
        input_dict['unscaled_penalty_loss'] = unscaled_penalty_loss
        input_dict['scaled_performance'] = fx + scaled_penalty_loss

        # perform Lagrangian update only during the training phase
        if self.training:
            if self.init:
                # initialize lagrange multipliers only at the beginning of the training
                self.lm = torch.zeros(self.nsamples, C.shape[-1])
                self.init = False
                self.penalty_best = torch.full((self.nsamples, 1), np.finfo(np.float32).max)

            if input_dict['epoch'] % self.inner_loop == 0 and input_dict['epoch'] != 0:
                """Then do outer loop"""
                # check which constraints improved to obtain binary flags in update_lm
                update_lm = penalties_sqrt < self.eta * self.penalty_best[input_dict['index']]
                # lm update for ineq constraints
                self.lm[input_dict['index']] = F.relu(self.lm[input_dict['index']] + \
                                                      self.mu[input_dict['index']] * C * update_lm)
                # update best penalty
                self.penalty_best[input_dict['index']] = self.penalty_best[input_dict['index']] * ~update_lm \
                                                         + penalties_sqrt * update_lm
                self.lm.detach_()
                sigma_update = ~update_lm * self.sigma + update_lm
                self.mu[input_dict['index']] = torch.clamp(sigma_update * self.mu[input_dict['index']], max=self.mu_max)

            mu_scaled_penalty_loss = torch.mean(self.mu * penalties)
            # con_lagrangian for ineq constraints
            con_lagrangian = torch.mean(torch.bmm(self.lm[input_dict['index']].unsqueeze(1), F.relu(C).unsqueeze(-1)))
            # calculate total augmented lagrangian loss
            input_dict['loss'] += con_lagrangian + mu_scaled_penalty_loss
            input_dict['con_lagrangian'] = con_lagrangian
            input_dict['mu_scaled_penalty_loss'] = mu_scaled_penalty_loss
            input_dict['mu'] = torch.mean(self.mu)
        else:
            input_dict['loss'] += scaled_penalty_loss

        return input_dict


losses = {'penalty': PenaltyLoss,
          'barrier': BarrierLoss,
          'augmented_lagrange': AugmentedLagrangeLoss}


def get_loss(objectives, constraints, train_data, args):
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints,
                           barrier=args.barrier_type)
    elif args.loss == 'augmented_lagrange':
        optimizer_args = {'inner_loop': args.inner_loop, "eta": args.eta, 'sigma': args.sigma,
                          'mu_init': args.mu_init, "mu_max": args.mu_max}
        loss = AugmentedLagrangeLoss(objectives, constraints, train_data, **optimizer_args)
    return loss
