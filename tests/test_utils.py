import pytest 
import unittest 
from neuromancer.utils import handle_device_placement
import torch 
import lightning.pytorch as pl

torch.manual_seed(0)

class DummyConstraint(pl.LightningModule):
    def __init__(self, callable):
        super().__init__()
        self.callable = callable

    @handle_device_placement
    def forward(self, left, right):
        return self.callable(left, right)

def callable_1(left, right):
    return left - right

def callable_2(left, right):
    return left + right

def callable_3(left, right):
    return left * right

@pytest.mark.parametrize("callable", [
    callable_1,
    callable_2,
    callable_3,
])
def test_handle_device_placement(callable):
    
    cons = DummyConstraint(callable)

    left_tensor = torch.randn(3, 3)  
    right_tensor = torch.randn(3, 3).to(device='cuda:1')  

    result = cons(left_tensor, right_tensor)

    if left_tensor.device.type == 'cpu' != right_tensor.device.type:
        expected_result = callable(left_tensor.type_as(right_tensor), right_tensor)
    elif right_tensor.device.type == 'cpu' != left_tensor.device.type:
        expected_result = callable(left_tensor, right_tensor.type_as(left_tensor))
    else:
        expected_result = callable(left_tensor, right_tensor)
        
    assert (torch.allclose(result, expected_result))
    assert result.device.type == 'cuda'
    assert result.get_device() == 1