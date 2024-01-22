import itertools
import torch
from neuromancer.constraint import variable
import pytest
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(True)


def test_init():
    variable()
    variable('x')
    variable(display_name="a")
    variable(3, 3)
    variable((3, 3))
    variable(torch.randn(3, 3, requires_grad=True))
    variable(3, 3, display_name="a")
    variable((3, 3), display_name="a")
    variable(torch.randn(3, 3, requires_grad=True), display_name="a")

    variable([variable(), variable()], lambda x, y: x + y)
    variable([variable()], torch.nn.Sequential(torch.nn.Linear(1, 1)))
    torch.add(variable('a'), torch.randn(1, 1))
    torch.add(variable('a'), variable('b'))
    with pytest.raises(Exception):
        variable(3, 3, is_input=True)
    with pytest.raises(Exception):
        variable((3, 3), is_input=True)
    with pytest.raises(Exception):
        variable(torch.randn(3, 3, requires_grad=True), is_input=True)
    with pytest.raises(Exception):
        variable(3, 3, display_name="a", is_input=True)
    with pytest.raises(Exception):
        variable((3, 3), display_name="a", is_input=True)
    with pytest.raises(Exception):
        variable(torch.randn(3, 3, requires_grad=True), display_name="a", is_input=True)


def test_equality():
    x = variable('x')
    assert x == x


def test_inequality():
    x = variable('x')
    y = variable('y')
    assert y != x


def test_inequality2():
    x = variable('x')
    y = variable('y')
    assert not y == x


def test_inclusion():
    x = variable('x')
    assert x in [x]
    assert x in {x}
    assert x in {x: 10}


def test_exclusion():
    x = variable('x')
    assert x not in [1, 2, 3]
    assert not x in {1, 2, 3}
    assert x not in {'x': 10}


def test_slicing():

    t1 = torch.ones(10, requires_grad=True)
    t2 = torch.ones(10, 10, requires_grad=True)
    t3 = torch.ones(10, 10, 10, requires_grad=True)
    d1 = variable(t1)
    d2 = variable(t2)
    d3 = variable(t3)
    assert torch.equal(d1[:5](), t1[:5])
    assert torch.equal(d2[:5, :5](), t2[:5, :5])
    assert torch.equal(d3[:5, :5, :5](), t3[:5, :5, :5])


def test_call():
    t1 = torch.ones(1)
    t2 = t1 * 10
    var_1 = variable('x')
    print(var_1._key)
    assert torch.equal(var_1({'x': t1}), t1)
    with pytest.raises(Exception):
        var_1()

    var_2 = var_1 * 10.0
    assert torch.equal(var_2({'x': t1}), t2)
    with pytest.raises(Exception):
        var_2()


def test_callables():

    t1 = torch.ones(1)

    a = variable([], lambda: t1)
    print(a())
    print(t1)
    assert torch.equal(a(), t1)
    b = variable([a], lambda x: 2 * x)
    assert torch.equal(b(), 2 * t1)
    c = variable([a, b], lambda x, y: x + y)
    assert torch.equal(c(), t1 + 2 * t1)


def test_arithmetic():

    v1 = torch.tensor((1.0,), requires_grad=True)
    v2 = torch.tensor((2.0,), requires_grad=True)

    a = variable(v1)
    b = variable(v2)

    assert torch.equal((a * b)(), v1 * v2)
    assert torch.equal((a + b)(), v1 + v2)
    # // a.k.a. '__floordiv__' is deprecated in torch
    assert torch.equal((a / b)(), v1 / v2)
    assert torch.equal((a @ b)(), v1 @ v2)
    assert torch.equal((a**b)(), v1**v2)


def test_builtins():
    v1 = torch.randn(1, requires_grad=True)
    a = variable(v1)

    assert torch.equal((-a)(), -v1)
    assert torch.equal(abs(a)(), abs(v1))
    # ~ a.k.a. '__invert__' only implemented for integers and booleans


def test_transpose():

    v1 = torch.randn(3, 3, requires_grad=True)
    a = variable(v1)

    assert torch.equal(a.T(), v1.T)
    assert torch.equal(a.mT(), v1.mT)


def test_torch_sum():
    t1 = torch.ones(100, requires_grad=True)
    a = variable(t1)
    b = torch.sum(a)
    assert torch.equal(b(), torch.sum(t1))


def test_norms():

    v1 = variable(torch.zeros(100, requires_grad=True))
    v2 = variable(torch.zeros(100, requires_grad=True))
    assert torch.equal(torch.linalg.norm(v1 - v2)(), torch.tensor(0.0))


def test_losses():

    funcs = [F.mse_loss, F.l1_loss, F.smooth_l1_loss, F.huber_loss]

    v1 = torch.randn(100, requires_grad=True)
    v2 = torch.randn(100, requires_grad=True)
    for f in funcs:
        l1 = f(v1, v2)
        l2 = f(variable(v1), variable(v2))()
        assert torch.equal(l1, l2)


def test_torch_module_interoperability():

    a = variable(torch.ones(10, 1, requires_grad=True))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 1))

        def forward(self, x):
            return self.net(x)

    net = Network()

    b = variable([a], net)
    b()


def test_torch_module_interoperability2():

    a = variable('a')
    net = nn.Sequential(nn.Linear(1, 1))
    b = variable([a], net)
    b({'a': torch.ones(1, 1)})


def test_torch_module_interoperability3():

    a = variable('a')
    b = torch.add(a, torch.ones(5, 5))
    assert torch.equal(b({'a': torch.ones(5, 5)}), 2.*torch.ones(5, 5))


def test_parameters():

    # for graphs containing a single node the value of 'recurse'
    # should not have any impact
    for recurse in [True, False]:
        a = variable()
        assert len(list(a.parameters(recurse=recurse))) == 1
        assert list(a.parameters(recurse=recurse))[0].shape == (1,)

        a = variable(3, 3)
        assert len(list(a.parameters(recurse=recurse))) == 1
        assert list(a.parameters(recurse=recurse))[0].shape == (3, 3)

        a = variable(torch.empty(3, 3, requires_grad=True))
        assert len(list(a.parameters(recurse=recurse))) == 1
        assert list(a.parameters(recurse=recurse))[0].shape == (3, 3)

        a = variable([], lambda: None)
        assert len(list(a.parameters(recurse=recurse))) == 0

        net = torch.nn.Sequential(torch.nn.Linear(2, 2))
        a = variable([], net)
        assert all(p in set(net.parameters()) for p in a.parameters(recurse=recurse))

        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        a = variable([], net)
        assert all(p in set(net.parameters()) for p in a.parameters(recurse=recurse))

    # when dealing with graphs of more than one node setting 'recurse_graph' to true
    a = variable()
    b = variable()
    c = a + b
    assert len(list(c.parameters())) == 2
    assert all(
        p in list(c.parameters())
        for p in itertools.chain(a.parameters(), b.parameters())
    )


def test_unpack():

    v = variable([], lambda: (0, 1, 2))

    a, b, c = v.unpack(3)

    assert a() == 0
    assert b() == 1
    assert c() == 2


def test_unpack2():
    m = variable(torch.eye(10, 10))
    u, s, v = torch.linalg.svd(m).unpack(["u", "s", "v"])
    assert torch.equal(u(), torch.eye(10, 10))
    assert torch.equal(s(), torch.ones(10))
    assert torch.equal(v(), torch.eye(10))


def test_gradient():
    indict = {'x': torch.zeros(2,2, requires_grad=True)}
    x = variable('x')
    matmul = x @ x.T
    value = torch.sum(matmul)
    grad = value.grad(x)
    assert torch.equal(grad(indict), indict['x'])