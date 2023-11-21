import torch
import torch.nn as nn
from collections import OrderedDict
from abc import ABC, abstractmethod

######################## AGENT DEFINITIONS AND SPECIFICATIONS ######################

class Agent(nn.Module, ABC):
    """
    An agent is an object in a networked or otherwise distributed system that can:
    - have some intrinsic physics
    - serve as anchor for connections (pins)
    """

    def __init__(self, state_keys):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict
        """
        super().__init__()
        self.state_keys = state_keys

    @abstractmethod
    def intrinsic(self, x, y):
        """Calcuation of the intrinsic physics contribution from a particular agent

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent
        """
        pass

    @abstractmethod
    def algebra(self, x):
        """Algebraic update of algebraic states, if any.

        :param x: (torch.Tensor) input tensor of size (batches,in_keys), these are inputs to the algebra solver for the agent's algebraic states
        """
        pass

    def forward(self, x, y, mode: str = "ode"):
        """_summary_

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent
        :param mode: mode of the forward pass, "ode" or "dae", defaults to "ode"
        :raises ValueError: invalid mode
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        if mode == "ode":
            return self.intrinsic(x,y)
        elif mode == "dae":
            return self.algebra(x)
        else:
            raise ValueError("No match for ode or dae.")

### Children:
   
class SIMOConservationNode(Agent):
    """Single Input, Multiple Output Conservation Node. Useful for splitting mass flows, etc...
    """

    def __init__(self, state_keys = None, 
                in_keys = None, 
                solver = None):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        """
        super().__init__(state_keys=state_keys)
        self.solver = solver
        self.in_keys = in_keys

    def intrinsic(self, x, y):
        """No intrinsic physics contribution from conservation node, return zeros of correct size.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent        
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return torch.zeros_like(x[:,:len(self.state_keys)])

    def algebra(self, x):
        """Algebraic update for splitting single input among multiple outputs.

        :param x: (torch.Tensor) input tensor of size (batches,in_keys)
        :return: (torch.Tensor) output tensor of size (batches,state_keys)
        """

        param = torch.abs(self.solver(x[:,1:]))
        return torch.cat((x[:,[0]]*param,x[:,[0]]*(1.0 - param)),-1)
        #return x[:,[0]]*self.solver(x[:,1:])

class SIMOBBConservationNode(Agent):
    """Single Input, Multiple Output Conservation Node. Useful for splitting mass flows, etc...
    """

    def __init__(self, state_keys = None, 
                in_keys = None, 
                solver = None):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        """
        super().__init__(state_keys=state_keys)
        self.solver = solver
        self.in_keys = in_keys

    def intrinsic(self, x, y):
        """No intrinsic physics contribution from conservation node, return zeros of correct size.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent        
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return torch.zeros_like(x[:,:len(self.state_keys)])

    def algebra(self, x):
        """Algebraic update for splitting single input among multiple outputs.

        :param x: (torch.Tensor) input tensor of size (batches,in_keys)
        :return: (torch.Tensor) output tensor of size (batches,state_keys)
        """

        return x[:,[0]]*self.solver(x[:,1:])

class SISOConservationNode(Agent):
    """Single input, single output conservation node. Useful for sources, sinks, drains, return mass flows, etc...
    """

    def __init__(self, 
                state_keys = None, 
                in_keys = None, 
                solver = None):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        """
        super().__init__(state_keys=state_keys)
        self.solver = solver
        self.in_keys = in_keys

    def intrinsic(self, x, y):
        """No intrinsic physics contribution from conservation node, return zeros of correct size.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent        
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return torch.zeros_like(x[:,:len(self.state_keys)])

    def algebra(self, x):
        """Algebraic update according to self.solver.

        :param x: (torch.Tensor) input tensor of size (batches,in_keys)
        :return: (torch.Tensor) output tensor of size (batches,state_keys)
        """
        return self.solver(x)

class MIMOTank(Agent):
    """Multiple Input, Multiple Output Conservation Node."""
    def __init__(self, profile = lambda x: 1.0,
                 state_keys = None,
                 in_keys = None,
                 scaling: float = 1.0):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        :param scaling: (float), optional scaling factor.
        :param solver: a callable mapping agent states to a state-derived property (e.g. area-height relationship), nn.Module or lambda function
        """
        
        super().__init__(state_keys = state_keys)
        self.profile = profile
        self.scaling = scaling
        self.in_keys = in_keys

    def intrinsic(self, x, y):
        """Time rate of change of amount of substance in fixed volume over time, equal to sum(mass flows)/area(height).

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent  
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        # need to sum all of the inlet and outlet contributions and then scale by the 'capacitance' of the agent
        #return torch.sum(y,1,keepdim=True)/self.profile(x)
        return self.scaling*y/self.profile(x)
    
    def algebra(self, x):
        """No algebraic update for agent's states.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return x[:,:len(self.state_keys)]

class BatchReactor(Agent):
    """Custom agent for an adiabatic batch reactor with exothermic chemical reactions."""

    def __init__(self, state_keys = None,
                 in_keys = None,
                 C = nn.Parameter(torch.tensor(1.0)),
                 kinetics = None):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        :param C: (nn.Parameter), thermal capacitance of the reactor.
        :param kinetics: (callable) surrogate model for chemical kinetics
        """
        
        super().__init__(state_keys = state_keys)
        self.in_keys = in_keys
        self.C = C
        self.kinetics = kinetics

    def intrinsic(self,x,y):
        """Time rate of change of temperature of reactor obeys a first-law energy balance.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent  
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return (1.0/self.C)*(1-(x[:,[3]]/(x[:,[1]]+x[:,[2]]+x[:,[3]])))*self.kinetics(x[:,[0]])
    
    def algebra(self, x):
        """No algebraic update for agent's states.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return x[:,:len(self.state_keys)]
    
class Reactions(Agent):
    """Custom agent for surrogate modeling of chemical reaction network. """

    def __init__(self, state_keys = None,
                 in_keys = None,
                 solver = None):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        """
        super().__init__(state_keys=state_keys)
        self.in_keys = in_keys
        self.solver = solver

    def intrinsic(self, x, y):
        """No intrinsic physics contribution from conservation node, return zeros of correct size.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent        
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return torch.zeros_like(x[:,:len(self.state_keys)])

    def algebra(self, x):
        """Algebraic update according to self.solver.

        :param x: (torch.Tensor) input tensor of size (batches,in_keys)
        :return: (torch.Tensor) output tensor of size (batches,state_keys)
        """
        return self.solver(x)
    

class RCNode(Agent):
    """
    RCNode agent. The intrinsic action of the agent is to effectively scale the interaction physics 
    according to the capacitance of the agent. Examples include lumped volumes, rooms, etc.
    """

    def __init__(self, state_keys = None,
                in_keys = None,    
                C = nn.Parameter(torch.tensor([1.0])), 
                scaling = 1.0):
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)) List of strings for input, defaults to None
        :param scaling: (float) scale factor. Useful for expeceted multi-scale physicsm defaults to 1.0
        :param C: (nn.Parameter) learnable capacitance, defaults to a value of 1.0
        """

        super().__init__(state_keys=state_keys)
        self.in_keys = in_keys
        self.C = C
        self.scaling = scaling

    def intrinsic(self, x, y):
        """Time rate of change of temperature of reactor obeys a first-law energy balance.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent  
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return torch.max(torch.tensor(1e-6),self.C)*self.scaling*y
    
    def algebra(self, x):
        """No algebraic update for agent's states.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return x[:,:len(self.state_keys)]

class SourceSink(Agent):
    """
    Generic Source / Sink agent. Useful for 'dummy' agents to which one can attach external signals.
    """

    def __init__(self, state_keys = None, 
                in_keys = None):        
        """
        :param state_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param in_keys: (list(Str)), List of strings for input, defaults to None
        :param solver: any callable mapping from in_keys to state_keys, e.g. nn.Module or lambda function, defaults to None
        """
        super().__init__(state_keys=state_keys)
        self.in_keys = in_keys

    def intrinsic(self, x, y):
        """No intrinsic physics contribution from conservation node, return zeros of correct size.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :param y: (torch.Tensor) input tensor of size (batches,agent states), these are the accumulated interactions at the agent        
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """        
        return torch.zeros_like(x[:,:len(self.state_keys)])
    
    def algebra(self, x):
        """No algebraic update for agent's states.

        :param x: (torch.Tensor) input tensor of size (batches,agent states), these are the states of the agent
        :return: (torch.Tensor) output tensor of size (batches,agent states)
        """
        return x[:,:len(self.state_keys)]


####################### COUPLING DEFINITIONS AND SPECIFICATIONS ####################

class Interaction(nn.Module, ABC):
    """
    An interaction is a physical connection between agents:
    - interactions are pairwise
    - interactions can be one-sided or symmetric (influence both agents)
    """

    def __init__(self, in_keys,
                pins: list[list[int]], 
                symmetric: bool = False):
        """
        :param in_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to None
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way or two-way interaction, default False
        """
        super().__init__()
        self.symmetric = symmetric
        self.in_keys = in_keys
        self.pins = pins

    @abstractmethod
    def interact(self, x):
        """Calculation of an interaction on an edge in a graph.

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        """
        pass

    def forward(self, x):
        """forward pass of the interaction.

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        :return: (torch.Tensor)
        """
        return self.interact(x)    

### Children: 
class Pipe(Interaction):
    """
    Imposition of a source term as an interaction.
    """
    def __init__(self,
                in_keys = [], 
                pins: list[list[int]] = [],
                symmetric: bool = True):
        """
        :param in_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to []]
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]]), defaults to []
        :param symmetric: one-way or two-way interaction, defaults to True
        """
        super().__init__(in_keys = in_keys, 
                        pins=pins, 
                        symmetric=symmetric)

    def interact(self, x):
        """Observation function. Return observed value from specified agent.

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        :return: (torch.Tensor)
        """
        return x[:,[0]]

class DeltaTemp(Interaction):
    """
    Interaction physics for difference in temperature (assumed) between agents.   
    """

    def __init__(self,
                in_keys = [],
                pins: list[list[int]] = [],
                R = nn.Parameter(torch.tensor(1.0)), 
                symmetric = False):
        """
        :param in_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to []]
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]]), defaults to []
        :param R: (nn.Parameter) 1/Resistance, learnable, defaults to 1.0
        :param symmetric: one-way or two-way interaction, defaults to True
        """

        super().__init__(in_keys = in_keys, 
                        pins=pins, 
                        symmetric=symmetric)
        self.R = R

    def interact(self, x):
        """calculation of temperature difference (or, abstractly, any state difference) between agents.
        Scales the temperature difference by (R = 1/Resistance).

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        :return: (torch.Tensor)
        """
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])

class DeltaTempSwitch(Interaction):
    """
    Interaction physics for difference in temperature (assumed) between agents. Switched on/off 
    depending on agent values (zero or nonzero).
    """
    def __init__(self,
                in_keys = [],
                pins: list[list[int]] = [],
                R = nn.Parameter(torch.tensor(1.0)), 
                symmetric = False):
        """
        :param in_keys: (list(Str)) List of strings corresponding to keys in the ODE system's state dict, defaults to []]
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]]), defaults to []
        :param R: (nn.Parameter) 1/Resistance, learnable, defaults to 1.0
        :param symmetric: one-way or two-way interaction, defaults to True
        """

        super().__init__(in_keys = in_keys, 
                        pins=pins, 
                        symmetric=symmetric)
        self.R = R

    def interact(self, x):
        """calculation of temperature difference (or, abstractly, any state difference) between agents.
        Scales the temperature difference by (R = 1/Resistance).
        Returns a minimum value (default is 1e-2) if paired agent's state value is zero.

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        :return: (torch.Tensor)
        """
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])*(x[:,[1]]>=0.0)

class HVACConnection(Interaction):
    """
    Imposition of a source term as an interaction.
    """
    def __init__(self,
                in_keys = [],
                pins: list[list[int]] = [],
                symmetric = False):
        """
        :param feature_name: (str) Specification of correct state for interaction physics
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way ot two-way interaction
        """
        super().__init__(in_keys = in_keys, 
                        pins=pins, 
                        symmetric=symmetric)

    def interact(self, x):
        """Observation function. Return observed value from specified agent.

        :param x: (torch.Tensor) input tensor of shape (batches,in keys), these are the selected input states opon which the interaction operates
        :return: (torch.Tensor)
        """
        return x[:,[1]] 
    
agents = {'SIMOConservationNode': SIMOConservationNode,
          'SISOConservationNode': SISOConservationNode,
          'MIMOTank': MIMOTank,
          'BatchReactor': BatchReactor,
          'Reactions': Reactions,
          'RCNode': RCNode,
          'SourceSink': SourceSink}

couplings = {'Pipe': Pipe,
             'DeltaTemp': DeltaTemp,
             'DeltaTempSwitch': DeltaTempSwitch,
             'HVACConnection': HVACConnection}