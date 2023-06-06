"""
Definition of neuromancer.Bound class as a neuromancer.Component to allow
imposing hard bound constraints in the forms:
xmin <= x <= xmax
"""

from typing import List
from abc import abstractmethod
import torch
from neuromancer.component import Component


class Bounds(Component):
    DEFAULT_INPUT_KEYS: List[str]

    def __init__(self, input_key_map={}, output_keys=[], name=None):
        """
        Bounds is canonical Component class for imposing hard bound constraints on input variable
        :param input_key_map: (dict {str: str}) mapping of canonical expected input keys to alternate keys
        :param output_keys: [str] optional string to define new variable key at the output
        :param name:
        """

        self._update_input_keys(input_key_map=input_key_map)
        self._update_output_keys(output_keys=output_keys)
        super().__init__(input_keys=self.input_keys, output_keys=self.output_keys, name=name)

    def _update_input_keys(self, input_key_map={}):
        assert isinstance(input_key_map, dict), \
            f"{type(self).__name__} input_key_map must be dict for remapping input variable names; "
        self.input_key_map = {
            **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_key_map.keys()},
            **input_key_map
        }
        self.input_keys = list(self.input_key_map.values())
        assert len(self.input_keys) == len(self.DEFAULT_INPUT_KEYS), \
            "Length of given input keys must equal the length of default input keys"

    def _update_output_keys(self, output_keys=[]):
        if bool(output_keys):
            self.output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
            assert len(output_keys) == 1, \
                f'output_keys must have only one element but {len(output_keys)} were given'
        elif bool(self.input_key_map):
            self.output_keys = [self.input_key_map[self.DEFAULT_INPUT_KEYS[0]]]
        else:
            self.output_keys = [self.DEFAULT_INPUT_KEYS[0]]

    @abstractmethod
    def forward(self, data):
        pass


class HardMinMaxScale(Bounds):
    DEFAULT_INPUT_KEYS = ["x", "xmin", "xmax"]
    """
    HardMinMaxScale imposes hard bound constraints on input variable of interest:
        xmin <= x_new <= xmax
        x_new = HardMinMaxScale(x)
    HardMinMaxScale cretes new variable x_new using sigmoid scaling of the original variable x
    thus the original values of x are not preserved
    use in situations where you care about the output variable x_new with smooth gradients
    """

    def __init__(self, input_key_map={}, output_keys=[], scaling=1., name=None):
        """
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys
            ["x", "xmin", "xmax"] to alternate keys, e.g., ["y", "ymin", "ymax"]
        :param output_keys: [str] optional string to define new variable key at the output,
            by default input string "x" is being used
        :param name:
        """

        super().__init__(input_key_map=input_key_map, output_keys=output_keys, name=name)
        self.scaling = scaling

    def forward(self, data):
        x = data[self.input_key_map['x']]
        xmin = data[self.input_key_map['xmin']]
        xmax = data[self.input_key_map['xmax']]
        x = (xmax - xmin) * torch.sigmoid(self.scaling*x) + xmin
        return {self.output_keys[0]: x}


class HardMinMaxBound(Bounds):
    DEFAULT_INPUT_KEYS = ["x", "xmin", "xmax"]
    """
    HardMinMaxBounds imposes hard bound constraints on input variable of interest: 
        xmin <= x <= xmax
    by using projection onto feasible set, thus preserving the original values of x
    """

    def __init__(self, input_key_map={}, output_keys=[], name=None):
        """
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys
            ["x", "xmin", "xmax"] to alternate keys, e.g., ["y", "ymin", "ymax"]
        :param output_keys: [str] optional string to define new variable key at the output,
            by default input string "x" is being used
        :param name:
        """
        super().__init__(input_key_map=input_key_map, output_keys=output_keys, name=name)

    def forward(self, data):
        x = data[self.input_key_map['x']]
        xmin = data[self.input_key_map['xmin']]
        xmax = data[self.input_key_map['xmax']]
        x = x + torch.relu(-x + xmin)
        x = x - torch.relu(x - xmax)
        return {self.output_keys[0]: x}


class HardMinBound(Bounds):
    DEFAULT_INPUT_KEYS = ["x", "xmin"]
    """
    HardMinBounds imposes hard minimum bound constraints on input variable of interest:
        xmin <= x
    by using projection onto feasible set, thus preserving the original values of x
    """

    def __init__(self, input_key_map={}, output_keys=[], name=None):
        """
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys
        ["x", "xmin"] to alternate keys, e.g., ["y", "ymin"]
        :param output_keys: [str] optional string to define new variable key at the output,
        by default input string "x" is being used
        :param name:
        """
        super().__init__(input_key_map=input_key_map, output_keys=output_keys, name=name)

    def forward(self, data):
        x = data[self.input_key_map['x']]
        xmin = data[self.input_key_map['xmin']]
        x = x + torch.relu(-x+xmin)
        return {self.output_keys[0]: x}


class HardMaxBound(Bounds):
    DEFAULT_INPUT_KEYS = ["x", "xmax"]
    """
    HardMaxBounds imposes hard maximum bound constraints on input variable of interest:
        x <= xmax
    by using projection onto feasible set, thus preserving the original values of x
    """

    def __init__(self, input_key_map={}, output_keys=[], scale=1., name=None):
        """
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys
        ["x", "xmax"] to alternate names, e.g., ["y", "ymax"]
        :param output_keys: [str] optional string to define new variable key at the output,
        by default input string "x" is being used
        :param scale: float scaling factor for input variable values, larger the value the tighter the binary approximation
        :param name:
        """
        super().__init__(input_key_map=input_key_map, output_keys=output_keys, name=name)
        self.scale = scale

    def forward(self, data):
        x = data[self.input_key_map['x']]
        xmax = data[self.input_key_map['xmax']]
        x = x - torch.relu(x - xmax)
        return {self.output_keys[0]: x}


