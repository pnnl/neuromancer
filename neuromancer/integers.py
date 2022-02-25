
"""
Definition of neuromancer.SoftBinary class and neuromancer.SoftInteger
imposing soft constraints on to approximate binary and integer variables
    x in Z
    x in N
"""

from abc import ABC, abstractmethod
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.component import Component


def sawtooth(value):
    # https: // en.wikipedia.org / wiki / Sawtooth_wave
    x = (torch.atan(torch.tan(np.pi * value))) / np.pi
    return x


def smooth_sawtooth(value):
    x = (torch.tanh(torch.tan(np.pi * value))) / np.pi
    return x


class VarConstraint(Component, ABC):

    def __init__(self, input_keys, output_keys=[], name=None):
        """
        VarConstraint is canonical Component class for imposing binary and integer constraints on variables in the input_keys
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        if bool(output_keys):
            output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
            assert len(output_keys) == len(input_keys), \
                f'output_keys must have the same number of elements as input_keys. \n' \
                f'{len(output_keys)} output_keys were given, but {len(input_keys)} were expected.'
        else:
            output_keys = input_keys
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)

    @abstractmethod
    def transform(self, x):
        pass

    def forward(self, data):
        output_dict = {}
        for key_in, key_out in zip(self.input_keys, self.output_keys):
            output_dict[key_out] = self.transform(data[key_in])
        return output_dict


class SoftBinary(VarConstraint):

    def __init__(self, input_keys, output_keys=[], threshold=0.0, scale=10., name=None):
        """
        SoftBinary is class for imposing soft binary constraints on input variables in the input_keys list
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param scale: float, scaling value for better conditioning of the soft binary approximation
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.scale = scale
        self.threshold = threshold

    def transform(self, x):
        return torch.sigmoid(self.scale*(x - self.threshold))


class IntegerCorrector(VarConstraint):
    step_methods = {'sawtooth': sawtooth,
                   'smooth_sawtooth': smooth_sawtooth}

    def __init__(self, input_keys, output_keys=[], method="sawtooth", nsteps=1, stepsize=0.5, name=None):
        """
        IntegerCorrector is class for imposing differentiable integer correction on input variables in the input_keys list
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param method: [str] integer correction step method ['sawtooth', 'smooth_sawtooth']
        :param nsteps:
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.step = self._set_step_method(method)
        self.nsteps = nsteps
        self.stepsize = stepsize

    def _set_step_method(self, method):
        if method in self.step_methods:
            return self.step_methods[method]
        else:
            assert callable(method), \
                f'The integer correction step method, {method} must be a key in {self.step_methods} ' \
                f'or a differentiable callable.'
            return method

    def transform(self, x):
        for k in range(self.nsteps):
            x = x - self.stepsize*self.step(x)
        return x

class BinaryCorrector(IntegerCorrector):

    def __init__(self, input_keys, output_keys=[], threshold=0.0, scale=1.,
                 method="sawtooth", nsteps=1, stepsize=0.5, name=None):
        """
        SoftBinary is class for imposing binary constraints correction on input variables in the input_keys list
        generates: x in {0, 1}
        if x <  threshold then x = 0
        if x >= threshold then x =1
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param scale: float, scaling value for better conditioning of the soft binary approximation
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, method=method,
                         nsteps=nsteps, stepsize=stepsize, name=name)
        self.scale = scale
        self.threshold = threshold

    def transform(self, x):
        x = torch.sigmoid(self.scale * (x - self.threshold))
        for k in range(self.nsteps):
            x = x - self.stepsize * self.step(x)
        return x


def generate_truth_table(num_binaries):
    table = list(itertools.product([0, 1], repeat=num_binaries))
    return torch.tensor(table)


def generate_MIP_con(num_binaries, tolerance=0.01):
    """
    Function for generating the coefficients of the mixed-integer (MIP) constraints A x_b >= b
    for encoding truth tables for binary variables x_b
    see Table 3.6 in: https://www.uiam.sk/assets/fileAccess.php?id=1305&type=1
    :param num_binaries: number of binary variables x_b
    :param tolerance: numerical tolerance for soft approximation of single binary variable x_b
    :return: A [tensor], b [tensor] as coefficients of the MIP constraints A x_b >= b
    """
    # A is defined by truth table given the number of binary variables num_binaries
    A = torch.tensor(list(itertools.product([0, 1], repeat=num_binaries)))
    A[A == 0] = -1
    # b is the right hand side of the generated MIP inequality A >= b
    b = A.sum(dim=1) - num_binaries*tolerance
    return A, b


def binary_encode_integer(min, max):
    """
    Function to generate binary encoding of an integer in the interval [min, max]
    generates coefficients A, b, of the mixed-integer (MIP) constraints A x_b >= b
    and corresponding vector integer values as tensors
    :param min:
    :param max:
    :return:
    """
    int_min = int(np.ceil(min))
    int_max = int(np.floor(max))
    int_range = int_max - int_min +1
    if int_range > 1:
        num_binaries = int(np.ceil(np.log2(int_range)))
        A, b = generate_MIP_con(num_binaries, tolerance=0.01)
        A = A[:int_range,:].float()
        b = b[:int_range]
        int_values = torch.arange(int_min, int_max+1)
        return A, b, int_values
    else:
        print('There is only one possible integer value to be encoded.')
        print(f'For this purpose use SoftBinary class for switching between 0 '
              f'and desired integer value x in the range {min} <= x <= {max}.')


def soft_binary_to_integer(binary, A, b, int_values, tolerance=0.01):
    """
    Converts the soft binary tensor into integer tensor
        soft_index = min(ReLU(b-A*binary), 1)
        soft_index should have only one nonzero element,
        bit this condition might be violated due to constraints tolerances
        alternative encoding for single hot encoding:
        soft_index = softmax(ReLU(b-A*binary)
        integer = int_values*soft_index
        for inference
        integer = int_values*ceil(soft_index)

    :param binary: Tensor, soft approximation of the binary variable
    :param A: Tensor, matrix of the MIP constraints
    :param b: Tensor, right hand side of the MIP constraints
    :param int_values: Tensor,  admissible values of the integer variables
            given by binary encoding via truth tables
    :param tolerance: numerical tolerance for soft approximation of single binary variable x_b
    :return: Tensor, integer valued tensor
    """
    offset = torch.ceil(F.relu(b-torch.matmul(A, binary)))*binary.shape[0]*tolerance
    soft_index = F.relu(b-torch.matmul(A, binary)) + offset
    int_values_masked = int_values*(soft_index)
    idx = (int_values_masked != 0)
    soft_integer = int_values_masked[idx]
    integer = int_values[idx]
    return soft_integer, integer



