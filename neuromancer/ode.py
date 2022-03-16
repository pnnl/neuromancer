import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ODESystem(nn.Module, ABC):
    """
    Class for defining RHS of arbitrary ODE functions, 
    can be mix-and-matched according to expert knowledge.
    """

    def __init__(self, insize, outsize):
        super().__init__()
        self.in_features, self.out_features = insize, outsize

    @abstractmethod
    def ode_equations(self, x):
        pass

    def forward(self, x):
        assert len(x.shape) == 2
        return self.ode_equations(x)


class TwoTankParam(ODESystem):
    def __init__(self, insize=4, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.c1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
    
    def ode_equations(self, x): 
        # heights in tanks
        h1 = torch.abs(x[:, [0]])    # (# batches,1)
        h2 = torch.abs(x[:, [1]])
        # Inputs (2): pump and valve
        pump = x[:, [2]]
        valve = x[:, [3]]
        # equations
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * torch.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * torch.sqrt(h1) - self.c2 * torch.sqrt(h2)
        return torch.cat([dhdt1, dhdt2], dim=-1)


class DuffingParam(ODESystem):
    def __init__(self, insize=3, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor([0.02]), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor([8.0]), requires_grad=False)
        self.omega = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        
    def ode_equations(self, x): 
        # heights in tanks
        x0 = x[:, [0]]  # (# batches,1)
        x1 = x[:, [1]]
        t =  x[:, [2]]
        # equations
        dx0dt = x1
        dx1dt = -self.delta*x1 - self.alpha*x0 - self.beta*x0**3 + self.gamma*torch.cos(self.omega*t)
        return torch.cat([dx0dt, dx1dt], dim=-1)


class BrusselatorParam(ODESystem):
    def __init__(self, insize=2, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
    
    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [-1]]
        dx1 = self.alpha + x2*x1**2 - self.beta*x1 - x1
        dx2 = self.beta*x1 - x2*x1**2
        return torch.cat([dx1, dx2], dim=-1)


class BrusselatorHybrid(ODESystem):
    def __init__(self, block, insize=2, outsize=2):
        """

        :param block:
        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.block = block
        self.alpha = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
        self.beta = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
        assert self.block.in_features == 2
        assert self.block.out_features == 1

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [-1]]
        dx1 = self.alpha + self.block(x) - self.beta*x1
        dx2 = -self.block(x)
        return torch.cat([dx1, dx2], dim=-1)


class LotkaVolterraHybrid(ODESystem):

    def __init__(self, block, insize=2, outsize=2):
        """

        :param block:
        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.block = block
        self.alpha = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.delta = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        assert self.block.in_features == 2
        assert self.block.out_features == 1

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [-1]]
        dx1 = self.alpha*x1 - self.beta*self.block(x)
        dx2 = self.delta*self.block(x) - self.gamma*x2
        return torch.cat([dx1, dx2], dim=-1)


class LotkaVolterraParam(ODESystem):

    def __init__(self, insize=2, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        
    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [-1]]
        dx1 = x1 - 0.1*x1*x2
        dx2 = 1.5*x2 + 0.75*0.1*x1*x2
        return torch.cat([dx1, dx2], dim=-1)


class LorenzParam(ODESystem):
    def __init__(self, insize=3, outsize=3):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.rho = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
        self.sigma = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
        self.beta = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [1]]
        x3 = x[:, [-1]]
        dx1 = self.sigma*(x2 - x1)
        dx2 = x1*(self.rho - x3) - x2
        dx3 = x1*x2 - self.beta*x3
        return torch.cat([dx1, dx2, dx3], dim=-1)


ode_param_systems_auto = {'LorenzParam': LorenzParam,
                          'LotkaVolterraParam': LotkaVolterraParam,
                          'BrusselatorParam': BrusselatorParam}

ode_param_systems_nonauto = {'DuffingParam': DuffingParam,
                             'TwoTankParam': TwoTankParam}

ode_hybrid_systems_auto = {'LotkaVolterraHybrid': LotkaVolterraHybrid,
                           'BrusselatorHybrid': BrusselatorHybrid}
