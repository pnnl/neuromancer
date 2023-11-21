import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SSM(nn.Module):
    """
    Baseline class for (neural) state space model (SSM)
    Implements discrete-time dynamical system:
        x_k+1 = fx(x_k) + fu(u_k) + fd(d_k)
    with variables:
        x_k - states
        u_k - control inputs
        d_k - disturbances

    """
    def __init__(self, fx, fu, nx, nu, fd=None, nd=0):
        """

        :param fx: (nn.Module) state transition dynamics
        :param fu: (nn.Module) input dynamics
        :param nx: (int) number of states
        :param nu: (int) number of inputs
        :param fd: (nn.Module) disturbance dynamics
        :param nd: (int) number of disturbances
        """
        super().__init__()
        self.fx, self.fu, self.fd = fx, fu, fd
        self.nx, self.nu, self.nd = nx, nu, nd
        self.in_features, self.out_features = nx+nu+nd, nx

    def forward(self, x, u, d=None):
        """

        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nu])
        :param d: (torch.Tensor, shape=[batchsize, nd])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        assert len(x.shape) == 2
        assert len(u.shape) == 2

        # state space model
        x = self.fx(x) + self.fu(u)

        # add disturbance dynamics
        if self.fd is not None and d is not None:
            assert len(d.shape) == 2
            x += self.fd(d)
        return x


class ODESystem(nn.Module, ABC):
    """
    Class for defining RHS of arbitrary ODE functions, 
    can be mix-and-matched according to expert knowledge.
    """

    def __init__(self, insize, outsize):
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nx = outsize
        self.nu = insize - outsize

    @abstractmethod
    def ode_equations(self, x, *args):
        pass

    def forward(self, x, *args):
        assert len(x.shape) == 2
        return self.ode_equations(x, *args)

class GeneralNetworkedODE(ODESystem):
    """ Coupled nonlinear dynamical system with heterogeneous agents. Can be used standalone for networked ODE systems with
    homo/heterogeneous agents or together with :class:'GeneralNetworkedAE' for the specification of differential-algebraic
    equations."""

    def __init__(self, states = None, 
                agents = None, 
                couplings = None,
                insize = None,
                outsize = None,
                ):
        """Constructor method.
        :param states: (dict) dictionary of state and index pairs, defaults to None
        :param agents: (list(Agents)) list of agents for the networked system, defaults to None
        :param couplings: (list(Couplings)) list of couplings between agents
        :param insize: (int) in state dimension, defaults to None
        :param outsize: (int) out state dimension, defaults to None
        """

        super().__init__(insize=insize, outsize=outsize)
        
        # Composition of network:
        self.states = states
        self.agents = nn.ModuleList(agents)
        self.couplings = nn.ModuleList(couplings)
        self.insize = insize
        self.outsize = outsize
 
    def ode_equations(self, x, *args):
        """Forward pass of the method. 

        :param x: (torch.Tensor) input tensor of size (batches,states)
        :return: (torch.Tensor) output tensor of size (batches,states)
        """

        # construct the RHS of the ODE via physics(states,accumulated interactions)[return only autonomous part]
        # dx/dt = f(x,\sum_A(couplings))
        return self.intrinsic_physics(x,self.coupling_physics(x, *args), *args)[:,:self.outsize]

    def intrinsic_physics(self, x, interactions, *args):
        """Calculation of intrinsic physics contributions from agents. For each agent, 
        aggregate interactions and call the forward pass of the agent.

        :param x: (torch.Tensor) input tensor of size (batches,states), these are the states of the system
        :param interactions: (torch.Tensor) input tensor of size (batches,states), these are the accumulated interactions of the system
        :return: (torch.Tensor) output tensor of size (batches,states)
        """
        features = torch.cat([x, *args], dim=-1)
        dx = torch.zeros_like(features)

        for agent in self.agents:
            """Get all of the relevant attributes from the agent object and pass them as arguments to agents() fwd pass. 
            Loop through each agent. construct RHS of the ODE.
            """

            state_idxs = list(map(self.states.get, agent.state_keys))
            input_idxs = list(map(self.states.get, agent.in_keys))
            dx[:,state_idxs] = agent(features[:,input_idxs], interactions[:,state_idxs])

        return dx
    
    def coupling_physics(self, x, *args):
        """Calculate aggregated coupling physics across all connections in self.couplings.

        :param x: (torch.Tensor) input tensor of size (batches,states), these are the states of the system
        :return: (torch.Tensor) output tensor of size (batches,states)
        """

        features = torch.cat([x, *args], dim=-1)
        dx = torch.zeros_like(features)
        
        # first loop over coupling physics listed in self.couplings
        for physics in self.couplings:
            # for each physics in self.couplings, loop over the pins and add contribution to dx
            for pin in physics.pins:
                
                send = list(map(self.states.get, self.agents[pin[0]].state_keys))
                receive = list(map(self.states.get, self.agents[pin[1]].state_keys))
                interaction_idx = list(map(self.states.get, physics.in_keys))

                contribution = physics(features[:,interaction_idx]) # -> x[:,[1]] - x[:,[0]]
                dx[:,receive] += contribution
                if physics.symmetric:
                    dx[:,send] -= contribution
        return dx 

class GeneralNetworkedAE(ODESystem):
    """General Networked Algebraic Equation class. This is an extension of the ODESystem class
    for handling update of algebraic states for differential-algebraic equations. Intended to 
    be used in conjunction with GeneralNetworkedODE class.   
    """
    
    def __init__(self, states = None,
                agents = None,
                insize = None,
                outsize = None):
        """Constructor method.
        :param states: (dict) dictionary of state and index pairs, defaults to None
        :param agents: (list(Agents)) list of agents for the networked system, defaults to None
        :param insize: (int) in state dimension, defaults to None
        :param outsize: (int) out state dimension, defaults to None
        """
        super().__init__(insize=insize, outsize=outsize)
        self.states = states
        self.agents = nn.ModuleList(agents)
        self.insize = insize
        self.outsize = outsize
       
    def ode_equations(self, x, *args):
        """Forward pass of the method. For evolution of algebraic states, we call 
        self.algebraic_equations(x) here.

        :param x: (torch.Tensor) input tensor of size (batches,states)
        :return: (torch.Tensor) output tensor of size (batches,states)
        """

        return self.algebraic_equations(x, *args)

    def algebraic_equations(self, x, *args):
        """Update of algebraic state variables according to agent-based algebra solvers or surrogates thereof.
    
        :param x: (torch.Tensor) input tensor of size (batches,states)
        :return: (torch.Tensor) output tensor of size (batches,states)
        """

        features = torch.cat([x, *args], dim=-1)
        dx = torch.clone(features)

        for agent in self.agents:
            # change the states at these indices based on the algebra solvers contained in the agents, if any
            dx[:,list(map(self.states.get, agent.state_keys))] = agent(features[:,list(map(self.states.get, agent.in_keys))], [], mode="dae") 

        return dx[:,:self.outsize]


class TwoTankParam(ODESystem):
    def __init__(self, insize=4, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.c1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
    
    def ode_equations(self, x, u):
        # heights in tanks
        h1 = torch.clip(x[:, [0]], min=0, max=1.0)
        h2 = torch.clip(x[:, [1]], min=0, max=1.0)
        # Inputs (2): pump and valve
        pump = torch.clip(u[:, [0]], min=0, max=1.0)
        valve = torch.clip(u[:, [1]], min=0, max=1.0)
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
        
    def ode_equations(self, x, u):
        # heights in tanks
        x0 = x[:, [0]]  # (# batches,1)
        x1 = x[:, [1]]
        t = u
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
        dx1 = self.alpha + self.block(x) - self.beta*x1 - x1
        dx2 = self.beta*x1 -self.block(x)
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

    def ode_equations(self, x, u):
        x1 = x[:, [0]]
        x2 = x[:, [1]]
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

    def ode_equations(self, x, u):
        x1 = x[:, [0]]
        x2 = x[:, [1]]
        x3 = x[:, [2]]
        u1 = u[:, [0]]
        u2 = u[:, [1]]
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

    def ode_equations(self, x, u):
        Ca = x[:, [0]]      # state: Concentration of A in CSTR (mol/m^3)
        T = x[:, [1]]       # state: Temperature in CSTR (K)
        Tc = u              # control: Temperature of cooling jacket (K)
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
                             'CSTR_Param': CSTR_Param,
                             'VanDerPolControl': VanDerPolControl}

ode_hybrid_systems_auto = {'LotkaVolterraHybrid': LotkaVolterraHybrid,
                           'BrusselatorHybrid': BrusselatorHybrid}

ode_networked_systems = {'GeneralNetworkedODE': GeneralNetworkedODE}

odes = {**ode_param_systems_auto,
        **ode_param_systems_nonauto,
        **ode_hybrid_systems_auto,
        **ode_networked_systems}