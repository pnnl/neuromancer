import neuromancer.constraint as cn

from hypothesis import given, settings, strategies as st
import torch



@given(constraint_class = st.sampled_from([cn.LT, cn.Eq, cn.GT]), device=st.sampled_from(['cpu', 'cuda:1']))
def test_constraint_device_placement(constraint_class, device):
    left_tensor = torch.tensor(10.)
    right_tensor = torch.tensor(7.)
    left_tensor = left_tensor.to(device=device)
    right_tensor = right_tensor.to('cpu')

    test_constraint = constraint_class()
    loss, value, penalty = test_constraint(left_tensor, right_tensor)
    if device == 'cuda:1':
        assert value.device.type == 'cuda'
        assert value.get_device() == 1
    else: 
        assert value.device.type == 'cpu'

    loss, value, penalty = test_constraint(right_tensor, left_tensor)
    if device == 'cuda:1':
        assert value.device.type == 'cuda'
        assert value.get_device() == 1
    else: 
        assert value.device.type == 'cpu'


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_add_two_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    z1 = x + y
    z2 = y + x
    data = {'x': torch.randn(shape), 'y': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_add_variable_tensor(shape):
    x = cn.variable('x')
    tensor = torch.randn(shape)
    z1 = x + tensor
    z2 = tensor + x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4),
       st.floats(-1., 1.))
@settings(max_examples=10, deadline=None)
def test_add_variable_float(shape, flt):
    x = cn.variable('x')
    z1 = x + flt
    z2 = flt + x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_subtract_two_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    z1 = x - y
    z2 = y - x
    data = {'x': torch.randn(shape), 'y': torch.randn(shape)}
    assert torch.equal(z1(data), -z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_subtract_variable_tensor(shape):
    x = cn.variable('x')
    tensor = torch.randn(shape)
    z1 = x - tensor
    z2 = tensor - x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), -z2(data)), \
        f'z1(data) = {z1(data)}, z2(data) = {z2(data)}'


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4),
       st.floats(-1., 1.))
@settings(max_examples=10, deadline=None)
def test_subtract_variable_float(shape, flt):
    x = cn.variable('x')
    z1 = x - flt
    z2 = flt - x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), -z2(data)), \
        f'z1(data) = {z1(data)}, z2(data) = {z2(data)}'


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_multiply_two_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    z1 = x * y
    z2 = y * x
    data = {'x': torch.randn(shape), 'y': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_multiply_variable_tensor(shape):
    x = cn.variable('x')
    tensor = torch.randn(shape)
    z1 = x * tensor
    z2 = tensor * x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4),
       st.floats(-1., 1.))
@settings(max_examples=10, deadline=None)
def test_multiply_variable_float(shape, flt):
    x = cn.variable('x')
    z1 = x * flt
    z2 = flt * x
    data = {'x': torch.randn(shape)}
    assert torch.equal(z1(data), z2(data))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_neg_variable(shape):
    x = cn.variable('x')
    negx = -x
    data = {'x': torch.randn(shape)}
    assert torch.equal(x(data), -negx(data))


def test_matmul_two_variables():
    x = cn.variable('x')
    y = cn.variable('y')
    z1 = x @ y
    data = {'x': torch.randn([2, 2]), 'y': torch.randn([2, 2])}
    assert torch.equal(z1(data), data['x'] @ data['y'])


def test_matmul_variable_tensor():
    x = cn.variable('x')
    xt = cn.variable('xt')
    tensor1 = torch.randn([2, 2])
    tensor2 = torch.randn([2, 2])
    z1 = x @ tensor2
    z2 = tensor2.transpose(0, 1) @ xt
    data = {'x': tensor1, 'xt': tensor1.transpose(0, 1)}
    assert torch.equal(z1(data), z2(data).transpose(0, 1))


@given(st.lists(st.integers(1, 10), min_size=1, max_size=4), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_power_two_variables(shape, power):
    x = cn.variable('x')
    data = {'x': torch.randn(shape), 'power': power}
    power = cn.variable('power')
    z = x**power
    assert torch.allclose(z(data), data['x']**torch.tensor(data['power']))


@given(st.lists(st.integers(1, 4), min_size=1, max_size=3), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_power_variable_integer(shape, power):
    x = cn.variable('x')
    data = {'x': torch.randn(shape)}
    z = x**power
    assert torch.allclose(z(data), data['x']**torch.tensor(power).to(data['x'].device))


@given(st.integers(1, 100), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_combine_ops_1(size, dims):
    x = cn.variable('x')
    y = cn.variable('y')
    z = cn.variable('z')
    xyz = x + y * z
    data = {'x': torch.randn([size for k in range(dims)]),
            'y': torch.randn([size for k in range(dims)]),
            'z': torch.randn([size for k in range(dims)])}
    assert(torch.equal(xyz(data), data['x'] + data['y'] * data['z']))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_divide_two_variables(shape, divisor):
    x = cn.variable('x')
    data = {'x': torch.randn(shape), 'divisor': divisor}
    z = x / divisor
    print(z(data), data['x'] / torch.tensor(data['divisor']).to(data['x'].device))
    assert torch.equal(z(data), data['x'] / torch.tensor(data['divisor']).to(data['x'].device))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4), st.floats(1, 4))
@settings(max_examples=10, deadline=None)
def test_divide_two_variables2(shape, divisor):
    x = cn.variable('x')
    data = {'x': torch.randn(shape, requires_grad=True), 'divisor': torch.tensor(divisor)}
    divisor = cn.variable(torch.tensor(divisor))
    z = x / divisor
    assert torch.equal(z(data), data['x'] / data['divisor'].to(data['x'].device))


@given(st.lists(st.integers(1, 2), min_size=1, max_size=2), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_divide_variable_integer(shape, divisor):
    x = cn.variable('x')
    data = {'x': torch.randn(shape)}
    z = x / divisor
    assert torch.equal(z(data), data['x'] / torch.tensor(divisor).to(data['x'].device))


@given(st.lists(st.integers(1, 2), min_size=1, max_size=2), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_divide_variable_integer2(shape, divisor):
    x = cn.variable('x')
    data = {'x': torch.randn(shape)}
    divisor = torch.tensor(divisor)
    z = x / divisor
    assert torch.equal(z(data), data['x'] / divisor.to(data['x'].device))


@given(st.integers(1, 100), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_combine_ops_1(size, dims):
    x = cn.variable('x')
    y = cn.variable('y')
    z = cn.variable('z')
    xyz = x + y * z
    data = {'x': torch.randn([size for k in range(dims)]),
            'y': torch.randn([size for k in range(dims)]),
            'z': torch.randn([size for k in range(dims)])}
    assert(torch.equal(xyz(data), data['x'] + data['y'] * data['z']))


@given(st.integers(1, 100), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_combine_ops_2(size, dims):
    x = cn.variable('x')
    y = cn.variable('y')
    z = cn.variable('z')
    xyz = x - y * z
    data = {'x': torch.randn([size for k in range(dims)]),
            'y': torch.randn([size for k in range(dims)]),
            'z': torch.randn([size for k in range(dims)])}
    assert(torch.equal(xyz(data), data['x'] - data['y'] * data['z']))


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_eq_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x == y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_eq_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x == y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_lt_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x < y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_lt_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x < y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_le_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x <= y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_le_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x <= y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_gt_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x > y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_gt_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x > y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ge_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x >= y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ge_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x >= y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] == 0.

####################################################################################

@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_neq_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x == y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_neq_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x == y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor +2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nlt_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x < y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nlt_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x < y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nle_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x <= y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nle_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x <= y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor - 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ngt_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x > y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ngt_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x > y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nge_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = x >= y
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nge_mse_variables(shape):
    x = cn.variable('x')
    y = cn.variable('y')
    cnstr = (x >= y)^2
    tensor = torch.randn(shape)
    data = {'x': tensor, 'y': tensor + 2}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_eq_number(shape):
    x = cn.variable(torch.tensor(1.))
    cnstr = x == 1.
    data = {}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_eq_mse_number(shape):
    x = cn.variable(torch.tensor(1.))
    cnstr = (1. == x)^2
    data = {}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_lt_number(shape):
    x = cn.variable('x')
    cnstr = x < 2.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_lt_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x < 2.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_le_number(shape):
    x = cn.variable('x')
    cnstr = x <= 2.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_le_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x <= 2.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_gt_number(shape):
    x = cn.variable('x')
    cnstr = x > -1.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_gt_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x > -1.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ge_number(shape):
    x = cn.variable('x')
    cnstr = x >= -1.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ge_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x >= -1)^2.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] == 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_neq_number(shape):
    x = cn.variable('x')
    cnstr = x == torch.rand(shape)
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_neq_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x == torch.rand(shape))^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nlt_number(shape):
    x = cn.variable('x')
    cnstr = x < -1.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nlt_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x < -1.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nle_number(shape):
    x = cn.variable('x')
    cnstr = x <= -1.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nle_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x <= -1.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ngt_number(shape):
    x = cn.variable('x')
    cnstr = x > 2.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_ngt_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x > 2.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nge_number(shape):
    x = cn.variable('x')
    cnstr = x >= 2.
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_nge_mse_number(shape):
    x = cn.variable('x')
    cnstr = (x >= 2.)^2
    data = {'x': torch.rand(shape)}
    assert cnstr(data)[cnstr.key] != 0.


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_variable_slicing(shape):
    x = cn.variable('x')
    data = {'x': torch.rand(shape)}
    assert torch.equal(x[1:](data), data['x'][1:])


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_variable_expression_slicing(shape):
    x = cn.variable('x')
    data = {'x': torch.rand(shape)}
    assert torch.equal((x+x)[1:](data), data['x'][1:] + data['x'][1:])


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_variable_expression_slicing2(shape):
    x = cn.variable('x')
    data = {'x': torch.rand(shape)}
    assert torch.equal((x[1:] + x[1:])(data), data['x'][1:] + data['x'][1:])


@given(st.lists(st.integers(1, 100), min_size=1, max_size=4))
@settings(max_examples=10, deadline=None)
def test_variable_expression_slicing_shape(shape):
    x = cn.variable('x')
    data = {'x': torch.rand(shape)}
    assert (x+x)[1:](data).shape[0] == (x + x)(data).shape[0] - 1


