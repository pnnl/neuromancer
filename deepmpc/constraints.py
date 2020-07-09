"""
constraints
used in model.regularize()

Types of constaints:
    1. Penalty functions
    # https: // en.wikipedia.org / wiki / Penalty_method
    some useful notes and theorems
    http://www-personal.umich.edu/~mepelman/teaching/NLP/Handouts/NLPnotes12_89.pdf

    admissible penalty functions: relu, relu6, softplus, SoftExponential,
    relu(g(x))^i where i>=1
        i=1 is a linear penalty function -  may not be di↵erentiable at g(x) = 0
        i=2 is a quadratic penalty - most common type

    2. one time step residual penalty
    # residual: dx_k = x_k- x_{k-1}
    admissible penalties: square, abs

    3. extension to exact penalty methods and augmented lagrangian method
    http://www-personal.umich.edu/~mepelman/teaching/NLP/Handouts/NLPnotes12_89.pdf
"""


# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
#local imports
import loops

# TODO: should we define the constraints as functions instead of modules?
# would make sense as they have no learnable parameters
# alternatively we could create a learnable variants where bounds would be parameters and not inputs
# TODO: motivation for learnable constraints - see research tangents
# e.g., reachability analysis given fixed model
# e.g., learning lipschitz constants for NN

class Penalty(nn.Module):
    def __init__(self, penalty=F.relu, weight=1.0):
        super().__init__()
        self.penalty = penalty
        self.weight = weight


class MinPenalty(Penalty):
    def __init__(self, penalty=F.relu, weight=1.0):
        """
        penalty on violating threshold inequality constraint:  x \ge xmin
        residual: s = penalty(-x + xmin)
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__(penalty, weight)

    def forward(self, x, xmin):
        return self.weight*self.penalty(-x + xmin)


class StaticMinPenalty(Penalty):
    def __init__(self, xmin, penalty=F.relu, weight=1.0):
        """
        penalty on violating threshold inequality constraint:  x \ge xmin
        residual: s = penalty(-x + xmin)
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__(penalty, weight)
        self.xmin = xmin

    def forward(self, x):
        return self.weight*self.penalty(-x + self.xmin)



class MaxPenalty(Penalty):
    def __init__(self, penalty=F.relu, weight=1.0):
        """
        penalty on violating threshold inequality constraint: x \le xmax
        residual: s = penalty(x - xmax)
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__(penalty, weight)

    def forward(self, x, xmax):
        return self.weight*self.penalty(x - xmax)


class MinMaxPenalty(Penalty):
    def __init__(self, penalty=F.relu, weight=1.0):
        """
        penalty on violating thresholds inequality constraints:
        x \ge xmin
        x \le xmax
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__(penalty, weight)

    def forward(self, x, xmin, xmax):
        return self.weight*self.penalty(-x + xmin) + self.penalty(x - xmax)


class EqualPenalty(Penalty):
    def __init__(self, penalty='square', weight=1.0):
        """
        penalty on equality constraint x = b
        residual: s = penalty(x - b)
        admissible penalties: square, abs
        """
        super().__init__(penalty, weight)

    def forward(self, x, b):
        if self.penalty == 'square':
            dx = (x - b) * (x - b)
        elif self.penalty == 'abs':
            dx = (x - b).abs()
        return self.weight*dx

class QuadraticPenalty(Penalty):
    def __init(self, penalty='quadratic', weight=1.0):
        super().__init__(penalty, weight)
        # https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
        # do we need this? or can we just use the min max constraints on the outputs of Polynomial block?


# TODO: do we want? higher level API for objective/constraints definition with pre-defined most commonly used types
def equals(x1, x2, weight=1.0, penalty=torch.nn.functional.mse_loss):
    """
    high level wrapper equality constraint x1 = x2

    :param x1: (dict, {str: torch.Tensor})
    :param x2: (dict, {str: torch.Tensor})
    :return: Objective object
    """
    if isinstance(x1, str):
        a = x1
    elif isinstance(x1, dict):
        a = list(x1.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    if isinstance(x2, str):
        b = x2
    elif isinstance(x2, dict):
        b = list(x2.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    expression = loops.Objective([a, b], penalty, weight=weight)
    return expression


def le(x, xmax, weight=1.0, penalty=F.relu, p=2):
    """
    high level wrapper less or equal then constraint: x \le xmax

    :param x: (dict, {str: torch.Tensor})
    :param xmax: (dict, {str: torch.Tensor})
    :param weight: weight of the penalty
    :param penalty: type of the penalty
    :param p: order of the penalty
    :return: Objective object
    """
    if isinstance(x, str):
        a = x
    elif isinstance(x, dict):
        a = list(x.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    if isinstance(xmax, str):
        b = xmax
    elif isinstance(xmax, dict):
        b = list(xmax.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    expression = loops.Objective([a, b], lambda x, xmax: weight * (penalty(x - xmax))**p, weight=weight)
    return expression

def ge(x, xmin, weight=1.0, penalty=F.relu, p=2):
    """
    high level wrapper greater or equal then constraint: x \ge xmin

    :param x: (dict, {str: torch.Tensor})
    :param xmax: (dict, {str: torch.Tensor})
    :param weight: weight of the penalty
    :param penalty: type of the penalty
    :param p: order of the penalty
    :return: Objective object
    """
    if isinstance(x, str):
        a = x
    elif isinstance(x, dict):
        a = list(x.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    if isinstance(xmin, str):
        b = xmin
    elif isinstance(xmin, dict):
        b = list(xmin.keys())[0]
    else:
        warnings.warn('argument must be string or dictionary')
    #     TODO: we need to pass this through mean
    expression = loops.Objective([a, b], lambda x, xmin: weight * (penalty(-x + xmin))**p, weight=weight)
    return expression


constraints = [MinPenalty, MaxPenalty, MinMaxPenalty, EqualPenalty]


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(N, samples, nx)
    x_prev = torch.rand(N, samples, nx)
    x_min = 0.25*torch.ones(N, samples, nx)
    x_max = 0.75 * torch.ones(N, samples, nx)

    xmin_con = MinPenalty()
    xmax_con = MaxPenalty()
    xminmax_con = MinMaxPenalty()
    dx_con1 = EqualPenalty()
    dx_con2 = EqualPenalty(penalty='abs')

    print(xmin_con(x, x_min).shape)
    print(xmax_con(x, x_max).shape)
    print(xminmax_con(x, x_min, x_max).shape)
    print(dx_con1(x, x_prev).shape)
    print(dx_con2(x, x_prev).shape)

    # testing new objective wrappers
    data = {'x': x, 'x_min': x_min, 'x_max': x_max}
    X = {'x': x}
    Xmin = {'x_min': x_min}
    Xmax = {'x_max': x_max}
    constr1 = le(X, Xmax)
    constr2 = le('x', 'x_max')
    constr3 = ge('x', 'x_min')
    constr4 = equals(X, X)
    # eval new constraints
    con1_val = constr1(data)
    con2_val = constr2(data)
    con3_val = constr3(data)
    con4_val = constr4(data)