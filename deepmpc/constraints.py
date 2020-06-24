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
#local imports


# TODO: should we define the constraints as functions instead of modules?
# would make sense as they have no learnable parameters
# alternatively we could create a learnable variants where bounds would be parameters and not inputs

class Penalty(nn.Module):
    def __init__(self, penalty=F.relu, weight=1.0):
        super().__init__()
        self.penalty = penalty
        self.weight = weight


class MinPenalty(Penalty):
    def __init__(self, penalty=F.relu, weight=1.0):
        """
        penalty on violating minimum threshold inequality constraint:  x \ge xmin
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__(penalty, weight)

    def forward(self, x, xmin):
        return self.weight*self.penalty(-x + xmin)


class MaxPenalty(Penalty):
    def __init__(self, penalty=F.relu, weight=1.0):
        """
        penalty on violating maximum threshold inequality constraint: x \le xmax
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


class dxPenalty(Penalty):
    def __init__(self, penalty='square', weight=1.0):
        """
        one time step residual penalty
        residual: dx_k = penalty(x_k- x_{k-1})
        admissible penalties: square, abs
        """
        super().__init__(penalty, weight)

    def forward(self, x, x_prev):
        if self.penalty == 'square':
            dx = (x - x_prev) * (x - x_prev)
        elif self.penalty == 'abs':
            dx = (x - x_prev).abs()
        return self.weight*dx


class QuadraticPenalty(Penalty):
    def __init(self, penalty='quadratic', weight=1.0):
        super().__init__(penalty, weight)


# https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
# do we need this? or can we just use the min max constraints on the outputs of Polynomial block?

constraints = [MinPenalty, MaxPenalty, MinMaxPenalty, dxPenalty]


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
    dx_con1 = dxPenalty()
    dx_con2 = dxPenalty(penalty='abs')

    print(xmin_con(x, x_min).shape)
    print(xmax_con(x, x_max).shape)
    print(xminmax_con(x, x_min, x_max).shape)
    print(dx_con1(x, x_prev).shape)
    print(dx_con2(x, x_prev).shape)