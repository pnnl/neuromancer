from hypothesis import given, settings, strategies as st

import torch
from neuromancer.maps import Map, ManyToMany, OneToOne
from neuromancer import blocks

_ = torch.set_grad_enabled(False)


def test_Map_no_name_output_keys():
    block = blocks.MLP(5, 4)
    data = {'x': torch.rand(500, 5)}
    func = Map(block, input_keys=['x'], output_keys=['fx'], name=None)
    output_keys = list(func(data).keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'fx'


def test_Map_output():
    block = blocks.MLP(5, 4)
    data = {'x': torch.rand(500, 5)}
    assert torch.all(block(data['x']) == (Map(block, input_keys=['x'],
                                              output_keys=['fx'], name=None)(data))['fx'])


def test_named_Map_output_keys():
    block = blocks.MLP(5, 4)
    data = {'x': torch.rand(500, 5)}
    func = Map(block, input_keys=['x'], output_keys=['fx'], name='mlp')
    output_keys = list(func(data).keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'fx'


def test_Map_no_name_output_keys_multi_input():
    block = blocks.MLP(10, 4)
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = Map(block, input_keys=['x', 'y'], output_keys=['fx'], name=None)
    output_keys = list(func(data).keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'fx'


def test_Map_output_multi_input():
    block = blocks.MLP(10, 4)
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    in_data = torch.cat([data['x'], data['y']], dim=-1)
    out = Map(block, input_keys=['x', 'y'], output_keys=['fx'], name=None)(data)
    assert torch.all(block(in_data) == out['fx'])


def test_named_Map_output_keys_multi_input():
    block = blocks.MLP(10, 4)
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = Map(block, input_keys=['x', 'y'], output_keys=['fx'], name='mlp')
    output_keys = list(func(data).keys())
    assert len(output_keys) == 1
    assert output_keys[0] == 'fx'


def test_OneToOne_no_name_output_keys():
    block = [blocks.MLP(5, 4), blocks.MLP(5, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = OneToOne(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name=None)
    output_keys = list(func(data).keys())
    assert len(output_keys) == 2
    assert output_keys[0] == 'fx'
    assert output_keys[1] == 'fy'


def test_OneToOne_output():
    block = [blocks.MLP(5, 4), blocks.MLP(5, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = OneToOne(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name=None)
    assert torch.all(block[0](data['x']) == func(data)['fx'])
    assert torch.all(block[1](data['y']) == func(data)['fy'])


def test_named_OneToOne_output_keys():
    block = [blocks.MLP(5, 4), blocks.MLP(5, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = OneToOne(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name='onetoone')
    output_keys = list(func(data).keys())
    assert len(output_keys) == 2
    assert output_keys[0] == 'fx'
    assert output_keys[1] == 'fy'


def test_ManyToMany_no_name_output_keys():
    block = [blocks.MLP(10, 4), blocks.MLP(10, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = ManyToMany(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name=None)
    output_keys = list(func(data).keys())
    assert len(output_keys) == 2
    assert output_keys[0] == 'fx'
    assert output_keys[1] == 'fy'


def test_ManyToMany_output():
    block = [blocks.MLP(10, 4), blocks.MLP(10, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = ManyToMany(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name=None)
    in_data = torch.cat([data['x'], data['y']], dim=-1)
    assert torch.all(block[0](in_data) == func(data)['fx'])
    assert torch.all(block[1](in_data) == func(data)['fy'])


def test_named_ManyToMany_output_keys():
    block = [blocks.MLP(10, 4), blocks.MLP(10, 4)]
    data = {'x': torch.rand(500, 5), 'y': torch.rand(500, 5)}
    func = ManyToMany(block, input_keys=['x', 'y'], output_keys=['fx', 'fy'], name='manytomany')
    output_keys = list(func(data).keys())
    assert len(output_keys) == 2
    assert output_keys[0] == 'fx'
    assert output_keys[1] == 'fy'
