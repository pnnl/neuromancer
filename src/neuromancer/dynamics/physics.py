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

    def __init__(self, state_names):
        super().__init__()
        self.state_names = state_names

    @abstractmethod
    def intrinsic(self, x):
        pass

    def forward(self,x):
        assert len(self.state_names) == x.shape[1]
        return self.intrinsic(x)

### Children:

class RCNode(Agent):
    """
    RCNode agent. The intrinsic action of the agent is to effectively scale the interaction physics 
    according to the capacitance of the agent. Examples include lumped volumes, rooms, etc.
    """

    def __init__(self, C = nn.Parameter(torch.tensor([1.0])), 
                state_names = ["T"],
                scaling = 1.0):
        """
        :param C: capacitance
        :param state_names: List of state names. Length should be same as dimension of state.
        :param scaling: scale factor. Useful for expeceted multi-scale physics.
        """

        super().__init__(state_names=state_names)
        self.C = C
        self.scaling = scaling

    def intrinsic(self, x):
        return torch.max(torch.tensor(1e-6),self.C)*self.scaling*x

class SourceSink(Agent):
    """
    Generic Source / Sink agent. Useful for 'dummy' agents to which one can attach external signals.
    """


    def __init__(self, state_names = ["T"]):
        """
        :param state_names: List of state names. Length should be same as dimension of state.
        """
        super().__init__(state_names = state_names)

    def intrinsic(self, x):
        return torch.zeros_like(x)

####################### COUPLING DEFINITIONS AND SPECIFICATIONS ####################

class Interaction(nn.Module, ABC):
    """
    An interaction is a physical connection between agents:
    - interactions are pairwise
    - interactions can be one-sided or symmetric (influence both agents)
    """

    def __init__(self, feature_name, pins, symmetric):
        """
        :param feature_name: (str) Specification of correct state for interaction physics
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way ot two-way interaction
        """
        super().__init__()
        self.symmetric = symmetric
        self.feature_name = feature_name
        self.pins = pins

    @abstractmethod
    def interact(self, x):
        pass

    def forward(self,x):
        return self.interact(x)    

### Children:

class DeltaTemp(Interaction):
    """
    Interaction physics for difference in temperature (assumed) between agents.   
    """

    def __init__(self,
                R = nn.Parameter(torch.tensor(1.0)), 
                feature_name = "T", 
                symmetric = False,
                pins = []):
        """
        :param R: resistivity for connection
        :param feature_name: (str) Specification of correct state for interaction physics
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way ot two-way interaction
        """

        super().__init__(feature_name=feature_name, pins=pins, symmetric=symmetric)
        self.R = R

    def interact(self, x):
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])

class DeltaTempSwitch(Interaction):
    """
    Interaction physics for difference in temperature (assumed) between agents. Switched on/off 
    depending on agent values (zero or nonzero).
    """
    def __init__(self,
                R = nn.Parameter(torch.tensor([1.0])), 
                feature_name = "T", 
                symmetric = False,
                pins = []):
        """
        :param R: resistivity for connection
        :param feature_name: (str) Specification of correct state for interaction physics
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way ot two-way interaction
        """

        super().__init__(feature_name=feature_name, pins=pins, symmetric=symmetric)
        self.R = R

    def interact(self, x):
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])*(x[:,[1]]>=0.0)

class HVACConnection(Interaction):
    """
    Imposition of a source term as an interaction.
    """
    def __init__(self,
                feature_name = "T",
                symmetric = False, 
                pins = []):
        """
        :param feature_name: (str) Specification of correct state for interaction physics
        :param pins: list of lists of pairwise connections between agents (e.g. pins=[[0,1],[0,2]])
        :param symmetric: one-way ot two-way interaction
        """
        super().__init__(feature_name=feature_name, pins=pins, symmetric=symmetric)

    def interact(self, x):
        return x[:,[1]] 

################################# HELPER FUNCTIONS #################################

def map_from_agents(intrinsic_list):
    """
    Quick helper function to construct state mappings:
    """
    
    agent_maps = []
    count = 0
    for agent_physics in intrinsic_list:
        node_states = [(s,i+count) for i,s in enumerate(agent_physics.state_names)]
        count += len(node_states)
        agent_maps.append(OrderedDict(node_states))

    return agent_maps

### Aggregate all in dicts:

agents = {'RCNode': RCNode,
          'SourceSink': SourceSink}

couplings = {'DeltaTemp': DeltaTemp,
             'DeltaTempSwitch': DeltaTempSwitch,
             'HVACConnection': HVACConnection}