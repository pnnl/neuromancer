import pytest
import torch
import torch.nn as nn
import neuromancer.slim as slim
import pydot
import warnings
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from collections import defaultdict
from hypothesis import given, settings, strategies as st

torch.manual_seed(0)

def components_example_1():
    # define neural architecture for the solution map
    func = blocks.MLP(insize=2, outsize=2,
                      bias=True,
                      linear_map=slim.maps['linear'],
                      nonlin=nn.ReLU,
                      hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class: sol_map(xi) -> x
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')
    # trainable components of the problem solution
    components = [sol_map]
    return components


def objectives_constraints_example_1():
    # define decision variables
    x1 = variable("x")[:, [0]]
    x2 = variable("x")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable('p')
    a = variable('a')

    # objective function
    f = (1 - x1) ** 2 + a * (x2 - x1 ** 2) ** 2
    obj = f.minimize(weight=1.0, name='obj')

    # constraints
    Q_con = 100.  # constraint penalty weights
    con_1 = Q_con * (x1 >= x2)
    con_2 = Q_con * ((p / 2) ** 2 <= x1 ** 2 + x2 ** 2)
    con_3 = Q_con * (x1 ** 2 + x2 ** 2 <= p ** 2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    return [obj], [con_1, con_2, con_3]

def get_edges_example_1():
    edges = defaultdict(list,
                {'in': ['map', 'map', 'obj', 'c2', 'c3'],
                 'map': ['obj', 'c1', 'c2', 'c3'],
                 'obj': ['out'],
                 'c1': ['out'],
                 'c2': ['out'],
                 'c3': ['out']})
    return dict(edges)


def dict_equals(dict1, dict2):

    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key == 'name':
            continue
        if key not in dict2:
            return False
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        if not torch.equal(tensor1, tensor2):
            return False
    return True


def get_test_dataloader():
    nsim = 5000  # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    a_low, a_high, p_low, p_high = 0.2, 1.2, 0.5, 2.0
    samples_train = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                     "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_dev = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                   "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_test = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                    "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    # create named dictionary datasets
    train_data = DictDataset(samples_train, name='train')
    dev_data = DictDataset(samples_dev, name='dev')
    test_data = DictDataset(samples_test, name='test')
    # create torch dataloaders for the Trainer
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, num_workers=0,
                                               collate_fn=train_data.collate_fn, shuffle=True)
    return train_loader


def step(test_data, problem):
    input_dict = test_data
    for node in problem.nodes:
        output_dict = node(input_dict)
        if isinstance(output_dict, torch.Tensor):
            output_dict = {node.name: output_dict}
        input_dict = {**input_dict, **output_dict}
    return input_dict


def list_equals_modulelist(lst, mod_list):
    lst2 = []
    for elem in mod_list:
        lst2.append(elem)
    return lst == lst2


def test_problem_initialization():
    components = components_example_1()
    objectives, constraints = objectives_constraints_example_1()
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

    #assert problem1.nodes == nn.ModuleList(components)
    assert problem.loss == loss
    assert problem.grad_inference == True
    assert problem.check_overwrite == True
    assert isinstance(problem, torch.nn.Module)


def test_problem_graph_generation():
    components = components_example_1()
    objectives, constraints = objectives_constraints_example_1()
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

    graph = problem.problem_graph
    assert graph is not None
    assert isinstance(graph, pydot.Dot)

    # Correct node names if they are unnamed
    input_keys = []
    output_keys = []
    nonames = 1
    for node in components:
        input_keys += node.input_keys
        output_keys += node.output_keys
        if node.name is None or node.name == '':
            node.name = f'node_{nonames}'
            nonames += 1

    expected_node_names = [node.name for node in components]  # expected_node_names
    for node in problem.nodes:  # node name after being ingested by system
        assert node.name in expected_node_names

    # Edge Testing -- determine if edges correctly instantiated
    edge_list = problem.problem_graph.get_edges()
    edges = defaultdict(list)
    for e in edge_list:
        src, dest = e.get_source(), e.get_destination()
        edges[src].append(dest)

    assert edges == get_edges_example_1()


def test_problem_step():
    components = components_example_1()
    objectives, constraints = objectives_constraints_example_1()
    loss = PenaltyLoss(objectives, constraints)
    test_dataloader = get_test_dataloader()
    test_data = next(iter(test_dataloader))

    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

    expected_output = step(test_data, problem)
    actual_output = problem.step(test_data)
    assert dict_equals(expected_output, actual_output)

def test_problem_loss():
    components = components_example_1()
    objectives, constraints = objectives_constraints_example_1()
    loss = PenaltyLoss(objectives, constraints)
    test_dataloader = get_test_dataloader()
    test_data = next(iter(test_dataloader))
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)
    output = problem(test_data)

    assert 'train_loss' in list(output.keys())

    loss_val = output['train_loss']
    assert loss_val.requires_grad



    def test_problem_initialization(self):
        components = components_example_1()
        objectives, constraints = objectives_constraints_example_1()
        loss = PenaltyLoss(objectives, constraints)
        problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

        assert problem is not None
        assert isinstance(problem.nodes, torch.nn.ModuleList)
        assert list_equals_modulelist(components, problem.nodes)
        assert problem.loss == loss
        assert problem.grad_inference == True
        assert problem.check_overwrite == True

    def test_problem_initialization_faulty(self):
        components = components_example_1()
        components += ["foo"]

        objectives, constraints = objectives_constraints_example_1()
        loss = PenaltyLoss(objectives, constraints)

        with pytest.raises(TypeError):
            Problem(components, loss)

        with pytest.raises(TypeError):
            Problem("Foo", loss)

        with pytest.raises(TypeError):
            problem2 = Problem(1, loss)

        with pytest.raises(TypeError):
            problem2 = Problem(nn.Module, loss)
"""
    def test_check_keys(self):
        components = components_example_1()
        node2 = Node(lambda x: x, ['p1'], ['y'], name='Node With Duplicate Key')

        objectives, constraints = objectives_constraints_example_1()
        loss = PenaltyLoss(objectives, constraints)

        problem1 = Problem(nodes=components, loss=loss, grad_inference=True, check_overwrite=True)
        with pytest.check_warnings():
            warnings.simplefilter("error")
            problem1._check_keys()

        components.append(node2)
        problem2 = Problem(nodes=components, loss=loss, grad_inference=True, check_overwrite=True)
        with pytest.warns():
            problem2._check_keys()

        problem2 = Problem(nodes=components, loss=loss, grad_inference=True, check_overwrite=False)
        with pytest.check_warnings():
            warnings.simplefilter("error")
            problem2._check_keys()

    def test_check_unique_names(self):
        # Test the _check_unique_names method
        # Create a Problem instance with nodes and loss that have unique names,
        # then call _check_unique_names and assert that no exceptions are raised.
        components = components_example_1()
        objectives, constraints = objectives_constraints_example_1()
        loss = PenaltyLoss(objectives, constraints)
        problem = Problem(nodes=components, loss=loss, grad_inference=True, check_overwrite=True)

        problem._check_unique_names()

        node2 = Node(lambda x: x, ['p1'], ['y'], name='map')
        new_components = components.append(node2)
        problem2 = Problem(nodes=new_components, loss=loss, grad_inference=True, check_overwrite=True)

        with pytest.raises(AssertionError):
            problem2._check_unique_names()
"""

