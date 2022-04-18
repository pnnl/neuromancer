from hypothesis import given, settings, strategies as st

import torch
from neuromancer.maps import Map, ManyToMany, OneToOne
from neuromancer.integers import SoftBinary, IntegerProjection, BinaryProjection
from neuromancer import blocks

_ = torch.set_grad_enabled(False)

methods = ["round_sawtooth", "round_smooth_sawtooth"]

def test_SoftBinary_IO_keys():
    data = {'x': torch.randn(500, 5)}
    bound1 = SoftBinary(input_keys=['x'])
    output_keys1 = list(bound1(data).keys())
    bound2 = SoftBinary(input_keys=['x'], output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_SoftBinary_multiIO_keys():
    data = {'x': torch.randn(500, 5), 'y': torch.randn(500, 5), 'z': torch.randn(500, 5)}
    bound1 = SoftBinary(input_keys=['x', 'y'])
    output_keys1 = list(bound1(data).keys())
    bound2 = SoftBinary(input_keys=['x', 'y', 'z'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 2
    assert output_keys1[0] == 'x'
    assert output_keys1[1] == 'y'
    assert len(output_keys2) == 3
    assert output_keys2[0] == 'x'
    assert output_keys2[1] == 'y'
    assert output_keys2[2] == 'z'


def test_SoftBinary_output():
    data = {'x': torch.randn(500, 5)}
    bound = SoftBinary(input_keys=['x'])
    out = bound(data)
    assert torch.all(out['x'] <= 1.0)
    assert torch.all(out['x'] >= 0.0)


def test_BinaryCorrector_IO_keys():
    data = {'x': torch.randn(500, 5)}
    bound1 = BinaryProjection(input_keys=['x'])
    output_keys1 = list(bound1(data).keys())
    bound2 = BinaryProjection(input_keys=['x'], output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_BinaryCorrector_multiIO_keys():
    data = {'x': torch.randn(500, 5), 'y': torch.randn(500, 5), 'z': torch.randn(500, 5)}
    bound1 = BinaryProjection(input_keys=['x', 'y'])
    output_keys1 = list(bound1(data).keys())
    bound2 = BinaryProjection(input_keys=['x', 'y', 'z'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 2
    assert output_keys1[0] == 'x'
    assert output_keys1[1] == 'y'
    assert len(output_keys2) == 3
    assert output_keys2[0] == 'x'
    assert output_keys2[1] == 'y'
    assert output_keys2[2] == 'z'


def test_BinaryCorrector_output():
    data = {'x': torch.randn(500, 5)}
    tolerance = 1e-6
    thresholds = [0.0, 0.5, 1.0, 2.0]
    for method in methods:
        for threshold in thresholds:
            nsteps = 1 if method == "round_sawtooth" else 5
            bound = BinaryProjection(input_keys=['x'], threshold=threshold, scale=1.,
                                    method=method, nsteps=nsteps, stepsize=1.0)
            out = bound(data)
            false_idx = data['x'] < threshold
            true_idx = data['x'] >= threshold
            assert torch.all(torch.abs(out['x'][false_idx]) <= tolerance)
            assert torch.all(torch.abs(out['x'][true_idx] - 1.0) <= tolerance)


def test_IntegerCorrector_IO_keys():
    data = {'x': torch.randn(500, 5)}
    bound1 = IntegerProjection(input_keys=['x'])
    output_keys1 = list(bound1(data).keys())
    bound2 = IntegerProjection(input_keys=['x'], output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_IntegerCorrector_multiIO_keys():
    data = {'x': torch.randn(500, 5), 'y': torch.randn(500, 5), 'z': torch.randn(500, 5)}
    bound1 = IntegerProjection(input_keys=['x', 'y'])
    output_keys1 = list(bound1(data).keys())
    bound2 = IntegerProjection(input_keys=['x', 'y', 'z'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 2
    assert output_keys1[0] == 'x'
    assert output_keys1[1] == 'y'
    assert len(output_keys2) == 3
    assert output_keys2[0] == 'x'
    assert output_keys2[1] == 'y'
    assert output_keys2[2] == 'z'


def test_IntegerCorrector_output():
    data = {'x': torch.randn(500, 5)}
    tolerance = 1e-6
    for method in methods:
        nsteps = 1 if method == "round_sawtooth" else 5
        bound = IntegerProjection(input_keys=['x'], method=method, nsteps=nsteps, stepsize=1.0)
        out = bound(data)
        assert torch.all(torch.abs(out['x'] - torch.round(data['x'])) <= tolerance)
        assert torch.all(torch.abs(out['x'] - torch.round(data['x'])) <= tolerance)
