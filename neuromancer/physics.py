import torch
import torch.nn as nn
from collections import OrderedDict
from abc import ABC, abstractmethod

######################## AGENT DEFINITIONS AND SPECIFICATIONS ######################

class Agent(nn.Module, ABC):

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

    def __init__(self, C = nn.Parameter(torch.tensor([1.0])), 
                state_names = ["T"],
                scaling = 1.0):

        super().__init__(state_names=state_names)
        self.C = C
        self.scaling = scaling

    def intrinsic(self, x):
        return torch.max(torch.tensor(1e-6),self.C)*self.scaling*x

class SourceSink(Agent):

    def __init__(self, state_names = ["T"]):
        super().__init__(state_names = state_names)

    def intrinsic(self, x):
        return torch.zeros_like(x)

####################### COUPLING DEFINITIONS AND SPECIFICATIONS ####################

class Interaction(nn.Module, ABC):

    def __init__(self, feature_name, pins, symmetric):
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

    def __init__(self,
                R = nn.Parameter(torch.tensor(1.0)), 
                feature_name = "T", 
                symmetric = False,
                pins = []):

        super().__init__(feature_name=feature_name, pins=pins, symmetric=symmetric)
        self.R = R

    def interact(self, x):
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])

class DeltaTempSwitch(Interaction):

    def __init__(self,
                R = nn.Parameter(torch.tensor([1.0])), 
                feature_name = "T", 
                symmetric = False,
                pins = []):

        super().__init__(feature_name=feature_name, pins=pins, symmetric=symmetric)
        self.R = R

    def interact(self, x):
        return torch.max(torch.tensor(1e-2),self.R)*(x[:,[1]] - x[:,[0]])*(x[:,[1]]>=0.0)

class HVACConnection(Interaction):

    def __init__(self,
                feature_name = "T",
                symmetric = False, 
                pins = []):

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