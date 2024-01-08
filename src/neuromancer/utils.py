

import functools
import lightning.pytorch as pl


def handle_device_placement(func):
    @functools.wraps(func)
    def wrapper(self, left, right):
        # Check if either tensor is on the CPU and the other on a non-CPU device
        if left.device.type == 'cpu' != right.device.type:
            left = left.type_as(right)
        elif right.device.type == 'cpu' != left.device.type:
            right = right.type_as(left)
        
        return func(self, left, right)
    
    return wrapper