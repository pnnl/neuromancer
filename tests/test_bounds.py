import torch
from neuromancer.modules.bounds import HardMinMaxScale, HardMinMaxBound, HardMinBound, HardMaxBound

_ = torch.set_grad_enabled(False)


def test_HardMinMaxScale_output_keys():
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5), 'xmax': torch.ones(500, 5)}
    bound1 = HardMinMaxScale()
    output_keys1 = list(bound1(data).keys())
    bound2 = HardMinMaxScale(output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_HardMinMaxScale_input_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1', 'xmax': 'xmax_1'}
    bound = HardMinMaxScale(input_key_map=input_key_map)
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_1'


def test_HardMinMaxScale_input_output_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1', 'xmax': 'xmax_1'}
    bound = HardMinMaxScale(input_key_map=input_key_map, output_keys=['x_new'])
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_new'


def test_HardMinMaxScale_output():
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5), 'xmax': torch.ones(500, 5)}
    bound = HardMinMaxScale()
    out = bound(data)
    assert torch.all(out['x'] <= data['xmax'])
    assert torch.all(out['x'] >= data['xmin'])


def test_HardMinMaxBound_output_keys():
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5), 'xmax': torch.ones(500, 5)}
    bound1 = HardMinMaxBound()
    output_keys1 = list(bound1(data).keys())
    bound2 = HardMinMaxBound(output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_HardMinMaxBound_input_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1', 'xmax': 'xmax_1'}
    bound = HardMinMaxBound(input_key_map=input_key_map)
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_1'


def test_HardMinMaxBound_input_output_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1', 'xmax': 'xmax_1'}
    bound = HardMinMaxBound(input_key_map=input_key_map, output_keys=['x_new'])
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_new'


def test_HardMinMaxBound_output():
    data = {'x': torch.randn(500, 5), 'xmin': -0.5*torch.ones(500, 5), 'xmax': 0.5*torch.ones(500, 5)}
    bound = HardMinMaxBound()
    out = bound(data)
    assert torch.all(out['x'] <= data['xmax'])
    assert torch.all(out['x'] >= data['xmin'])
    lt_index = data['x'] <= data['xmax']
    gt_index = data['x'] >= data['xmin']
    assert torch.all(out['x'][lt_index & gt_index] == data['x'][lt_index & gt_index])


def test_HardMinBound_output_keys():
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5)}
    bound1 = HardMinBound()
    output_keys1 = list(bound1(data).keys())
    bound2 = HardMinBound(output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_HardMinBound_input_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1'}
    bound = HardMinBound(input_key_map=input_key_map)
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_1'


def test_HardMinBound_input_output_keys():
    data = {'x_1': torch.randn(500, 5), 'xmin_1': -torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmin': 'xmin_1'}
    bound = HardMinBound(input_key_map=input_key_map, output_keys=['x_new'])
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_new'


def test_HardMinBound_output():
    data = {'x': torch.randn(100, 5), 'xmin': -0.5*torch.ones(100, 5)}
    bound = HardMinBound()
    out = bound(data)
    assert torch.all(out['x'] >= data['xmin'])
    gt_index = data['x'] >= data['xmin']
    assert torch.all(out['x'][gt_index] == data['x'][gt_index])


def test_HardMaxBound_output_keys():
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5), 'xmax': torch.ones(500, 5)}
    bound1 = HardMaxBound()
    output_keys1 = list(bound1(data).keys())
    bound2 = HardMaxBound(output_keys=['x1'])
    output_keys2 = list(bound2(data).keys())
    assert len(output_keys1) == 1
    assert output_keys1[0] == 'x'
    assert len(output_keys2) == 1
    assert output_keys2[0] == 'x1'


def test_HardMaxBound_input_keys():
    data = {'x_1': torch.randn(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1', 'xmax': 'xmax_1'}
    bound = HardMaxBound(input_key_map=input_key_map)
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_1'


def test_HardMaxBound_input_output_keys():
    data = {'x_1': torch.randn(500, 5), 'xmax_1': torch.ones(500, 5)}
    input_key_map = {'x': 'x_1','xmax': 'xmax_1'}
    bound = HardMaxBound(input_key_map=input_key_map, output_keys=['x_new'])
    out = bound(data)
    output_keys = list(out.keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'x_new'


def test_HardMaxBound_output():
    data = {'x': torch.randn(500, 5), 'xmax': 0.5*torch.ones(500, 5)}
    bound = HardMaxBound()
    out = bound(data)
    assert torch.all(out['x'] <= data['xmax'])
    lt_index = data['x'] <= data['xmax']
    assert torch.all(out['x'][lt_index] == data['x'][lt_index])
