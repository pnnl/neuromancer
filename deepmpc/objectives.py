"""
Objective functions

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: defines as functions or classes?

def Linear(x, c):
    """
    standard form linear loss
    """
    loss = torch.mm(c.T, x)
    return loss

def Quadratic(x, Q, c):
    """
    standard form quadratic loss
    """
    loss = 0.5*torch.mm(x.T, torch.mm(Q, x)) + torch.mm(c.T, x)
    return loss

def Nonlinear(x, fx):
    """
    user defined nonlinear loss
    """
    loss = fx(x)
    return loss


class CubicLoss(nn.Module):
    """
    example user defined cubic loss:  a*x**3 + b*x**2 + c*x
    """
    def __init__(self, nx, a, b, c):
        super().__init__()
        self.nx = nx
        self.a = a*torch.ones([1, nx])
        self.b = b*torch.ones([1, nx])
        self.c = c*torch.ones([1, nx])

    def forward(self, x):
        loss = torch.mm(self.a, x**3) + torch.mm(self.b, x**2) + torch.mm(self.c, x)
        return loss

# TODO: parser for high level symbolic definition of functions???

if __name__ == '__main__':
    nx = 3
    N = 5
    samples = 10
    X = torch.rand(N, samples, nx)
    x = X[:, 0, :].reshape(-1, 1)
    c = torch.rand(N*nx, 1)
    Q = torch.diag(torch.rand(N*nx))

    l_loss = Linear(x, c)
    q_loss = Quadratic(x, Q, c)

    # example custom loss
    a = 0.5
    b = 1
    c = 4
    fx = CubicLoss(N*nx, a, b, c)
    cubic_loss = fx(x)
