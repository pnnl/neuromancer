import pytest 
import unittest 
from neuromancer.utils import handle_device_placement
import torch 
import lightning.pytorch as pl

torch.manual_seed(0)

class DummyConstraint(pl.LightningModule):
    def __init__(self):
        super().__init__()

    @handle_device_placement
    def forward(self, left, right):
        value = left - right
        return value

def test_handle_device_placement():

    
    cons = DummyConstraint()


    left_tensor = torch.randn(3, 3)  
    right_tensor = torch.randn(3, 3).to(device='cuda:1')  


    result = cons(left_tensor, right_tensor)

    
    if left_tensor.device.type == 'cpu' != right_tensor.device.type:
        expected_result = left_tensor.type_as(right_tensor) - right_tensor
    elif right_tensor.device.type == 'cpu' != left_tensor.device.type:
        expected_result = left_tensor - right_tensor.type_as(left_tensor)
    else:
        expected_result = left_tensor - right_tensor 


    assert (torch.allclose(result, expected_result))
    assert result.device.type == 'cuda'
    assert result.get_device() == 1