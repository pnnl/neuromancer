import torch
from neuromancer.modules.functions import bounds_scaling, bounds_clamp

_ = torch.set_grad_enabled(False)


def test_bounds_clamp():
    x = torch.randn(500, 5)
    xmin = -torch.ones(500, 5)
    xmax = torch.ones(500, 5)
    out = bounds_clamp(x, xmin=xmin, xmax=xmax)
    assert torch.all(out <= xmax)
    assert torch.all(out >= xmin)


def test_bounds_scaling():
    x = torch.randn(500, 5)
    xmin = -torch.ones(500, 5)
    xmax = torch.ones(500, 5)
    out = bounds_scaling(x, xmin=xmin, xmax=xmax)
    assert torch.all(out <= xmax)
    assert torch.all(out >= xmin)


