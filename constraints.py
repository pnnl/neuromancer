"""
constraints
used in model.regularize()

Types of constaints:
    1. Penalty functions
    # https: // en.wikipedia.org / wiki / Penalty_method
    admissible penalty functions: relu, relu6, softplus, SoftExponential

    2. one time step residual penalty
    # residual: dx_k = x_k- x_{k-1}
    admissible penalties: square, abs

"""


# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import linear


class MinPenalty(nn.Module):
    def __init__(self, penalty=F.relu, **linargs):
        """
        penalty on violating minimum threshold inequality constraint:  x \ge xmin
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__()
        self.penalty = penalty

    def forward(self, x, xmin):
        return self.penalty(-x + xmin)

class MaxPenalty(nn.Module):
    def __init__(self, penalty=F.relu, **linargs):
        """
        penalty on violating maximum threshold inequality constraint: x \le xmax
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__()
        self.penalty = penalty

    def forward(self, x, xmax):
        return self.penalty(x - xmax)

class MinMaxPenalty(nn.Module):
    def __init__(self, penalty=F.relu, **linargs):
        """
        penalty on violating thresholds inequality constraints:
        x \ge xmin
        x \le xmax
        admissible penalty functions: relu, relu6, softplus, SoftExponential
        """
        super().__init__()
        self.penalty = penalty

    def forward(self, x, xmin, xmax):
        return self.penalty(-x + xmin) + self.penalty(x - xmax)

class dxPenalty(nn.Module):
    def __init__(self, penalty='square', **linargs):
        """
        one time step residual penalty
        residual: dx_k = penalty(x_k- x_{k-1})
        admissible penalties: square, abs
        """
        super().__init__()
        self.penalty = penalty

    def forward(self, x, x_prev):
        if self.penalty == 'square':
            dx = (x - x_prev) * (x - x_prev)
        elif self.penalty == 'abs':
            dx = (x - x_prev).abs()
        return dx



class QuadraticPenalty(nn.Module):
    pass
# https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
# do we need this? or can we just use the min max constraints on the outputs of Polynomial block?

constraints = [MinPenalty, MaxPenalty, MinMaxPenalty, dxPenalty]


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 40
    x = torch.rand(100, N, nx)
    x_prev = torch.rand(100, N, nx)
    x_min = 0.25*torch.ones(100, N, nx)
    x_max = 0.75*torch.rand(100, N, nx)

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