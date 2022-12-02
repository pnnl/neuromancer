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


class ControlODE(ODESystem):
    """
    Class for defining closed-loop dynamical system composed of ODEs and control policies,
    can be mix-and-matched according to expert knowledge.
    """

    def __init__(self, policy, ode, nx, nu, np=0, u_con=[]):
        """
        :param policy: (nn.Module) explicit control policy
        :param ode: (ODESystem or nn.Module) RHS of an ODE system
        :param nx: (int) number of state variables
        :param nu: (int) number of control input variables
        :param np: (int) number of control parameters
        :param u_con:
        """
        insize = nx + np
        outsize = nx
        super().__init__(insize=insize, outsize=outsize)
        self.nx, self.nu, self.np = nx, nu, np
        self.policy = policy
        self.ode = ode
        self.u_con = nn.ModuleList(u_con)
        assert isinstance(self.policy, nn.Module), \
            f'Control policy must be nn.Module, got {type(self.policy)}'
        assert isinstance(self.ode, ODESystem) or \
               isinstance(self.ode, nn.Module), \
            f'ODE must be ODESystem or nn.Module, got {type(self.ode)}'

    def reg_error(self):
        children_error = sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])
        control_penalty = 0
        for con in self.u_con:
            control_penalty += con({'u': self.u})[con.output_keys[0]]
        return children_error + control_penalty

    def ode_equations(self, xi):
        assert len(xi.shape) == 2, \
            f'Features must have two dimensions got {len(xi.shape)} dimensions'
        assert xi.shape[1] == self.nx + self.np, \
            f'Second feature dimension must be equal to nx ({self.nx}) + np ({self.np}), ' \
            f'got {xi.shape[1]}'
        u = self.policy(xi)
        self.u = u
        x = xi[:, :self.nx]
        xu = torch.cat([x, u], dim=-1)
        if isinstance(self.ode, ODESystem):
            dx = self.ode.ode_equations(xu)
        else:
            dx = self.ode(xu)
        return dx


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
        dhdt1 = torch.clamp(dhdt1, min=0, max=1.0)
        dhdt2 = torch.clamp(dhdt2, min=0, max=1.0)
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


class VanDerPolControl(ODESystem):

    def __init__(self, insize=3, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.mu = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [1]]
        u = x[:, [2]]
        dx1 = x2
        dx2 = self.mu*(1 - x1**2)*x2 - x1 + u
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


class LorenzControl(ODESystem):
    def __init__(self, insize=5, outsize=3):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.rho = torch.nn.Parameter(torch.tensor([28.0], requires_grad=True))
        self.sigma = torch.nn.Parameter(torch.tensor([10.0], requires_grad=True))
        self.beta = torch.nn.Parameter(torch.tensor([2.66667], requires_grad=True))

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [1]]
        x3 = x[:, [2]]
        u1 = x[:, [3]]
        u2 = x[:, [4]]
        dx1 = self.sigma * (x2 - x1) + u1
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3 - u2
        return torch.cat([dx1, dx2, dx3], dim=-1)


class CSTR_Param(ODESystem):
    def __init__(self, insize=3, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.nx = 2
        self.nu = 1
        # Volumetric Flowrate (m^3/sec)
        self.q = torch.nn.Parameter(torch.tensor([100.0], requires_grad=True))
        # Volume of CSTR (m^3)
        self.V = torch.nn.Parameter(torch.tensor([100.0], requires_grad=False))
        # Density of A-B Mixture (kg/m^3)
        self.rho = torch.nn.Parameter(torch.tensor([1000.0], requires_grad=True))
        # Heat capacity of A-B Mixture (J/kg-K)
        self.Cp = torch.nn.Parameter(torch.tensor([0.239], requires_grad=True))
        # Heat of reaction for A->B (J/mol)
        self.mdelH = torch.nn.Parameter(torch.tensor([5e4], requires_grad=False))
        # E - Activation energy in the Arrhenius Equation (J/mol)
        # R - Universal Gas Constant = 8.31451 J/mol-K
        self.EoverR = torch.nn.Parameter(torch.tensor([8750.], requires_grad=False))
        # Pre-exponential factor (1/sec)
        self.k0 = torch.nn.Parameter(torch.tensor([7.2e10], requires_grad=False))
        # U - Overall Heat Transfer Coefficient (W/m^2-K)
        # A - Area - this value is specific for the U calculation (m^2)
        self.UA = torch.nn.Parameter(torch.tensor([5e4], requires_grad=False))
        # Disturbances: Tf - Feed Temperature (K), Caf - Feed Concentration (mol/m^3)
        self.Tf = torch.nn.Parameter(torch.tensor([350.], requires_grad=False))
        self.Caf = torch.nn.Parameter(torch.tensor([1.], requires_grad=False))

    def ode_equations(self, x):
        Ca = x[:, [0]]      # state: Concentration of A in CSTR (mol/m^3)
        T = x[:, [1]]       # state: Temperature in CSTR (K)
        Tc = x[:, [2]]      # control: Temperature of cooling jacket (K)
        # reaction rate
        rA = self.k0 * torch.exp(-self.EoverR / T) * Ca
        # Calculate concentration derivative
        dCadt = self.q / self.V * (self.Caf - Ca) - rA
        # Calculate temperature derivative
        dTdt = self.q / self.V * (self.Tf - T) \
               + self.mdelH / (self.rho * self.Cp) * rA \
               + self.UA / self.V / self.rho / self.Cp * (Tc - T)
        return torch.cat([dCadt, dTdt], dim=-1)



ode_param_systems_auto = {'LorenzParam': LorenzParam,
                          'LotkaVolterraParam': LotkaVolterraParam,
                          'BrusselatorParam': BrusselatorParam}

ode_param_systems_nonauto = {'DuffingParam': DuffingParam,
                             'TwoTankParam': TwoTankParam,
                             'LorenzControl': LorenzControl,
                             'CSTR_Param': CSTR_Param}

ode_hybrid_systems_auto = {'LotkaVolterraHybrid': LotkaVolterraHybrid,
                           'BrusselatorHybrid': BrusselatorHybrid}
