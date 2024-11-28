import os
import random
import numpy as np
import torch
import functools
import lightning.pytorch as pl
from collections import OrderedDict

def handle_device_placement(func):
    """
    This is a decorator to handle automated GPU support for Neuromancer constraints. 
    It decorates a forward method that takes in two tensors (left and right) and ensures 
    both tensors reside on the same non-cpu device (if a GPU is available)
    """
    @functools.wraps(func)
    def wrapper(self, left, right):
        # Check if either tensor is on the CPU and the other on a non-CPU device
        if left.device.type == 'cpu' != right.device.type:
            left = left.type_as(right)
        elif right.device.type == 'cpu' != left.device.type:
            right = right.type_as(left)
        
        return func(self, left, right)
    
    return wrapper

def load_state_dict_lightning(problem, weight_path): 
    """
    This function handles loading problem weights when said problem was trained using 
    LitTrainer method

    :param problem: A Neuromancer Problem
    :param weight_path: (str) Path to weights saved by LitTrainer method
    """
    weights = torch.load(weight_path)['state_dict']
    weights = OrderedDict({key.replace('problem.', '', 1): value for key, value in weights.items()})
    problem.load_state_dict(weights)
    return problem

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
