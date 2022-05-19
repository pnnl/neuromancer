


import torch
import torch.nn as nn
from typing import List

from neuromancer.component import Component

# TODO: to implement closed loop component with recurrent policy
class ClosedLoop(Component):
    DEFAULT_SYSTEM_INPUT_KEYS: List[str]
    DEFAULT_POLICY_INPUT_KEYS: List[str]
    DEFAULT_SYSTEM_OUTPUT_KEYS: List[str]
    DEFAULT_POLICY_OUTPUT_KEYS: List[str]

    def __init__(self, input_key_map={}, name=None):
        pass
