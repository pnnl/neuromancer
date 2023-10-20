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


torch.manual_seed(0)

def example_1():
    """
    define an example 'problem' set-up, e.g. that from
    https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_1_basics.ipynb
    This will be used to test problem class
    """
    func = blocks.MLP(insize=2, outsize=2,
                      bias=True,
                      linear_map=slim.maps['linear'],
                      nonlin=nn.ReLU,
                      hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class: sol_map(xi) -> x
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')
    # trainable components of the problem solution
    components = [sol_map]

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

    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    loss = PenaltyLoss(objectives, constraints)

    edges = defaultdict(list,
                        {'in': ['map', 'map', 'obj', 'c2', 'c3'],
                         'map': ['obj', 'c1', 'c2', 'c3'],
                         'obj': ['out'],
                         'c1': ['out'],
                         'c2': ['out'],
                         'c3': ['out']})
    edges = dict(edges)

    return objectives, constraints, components, loss, edges


def dict_equals(dict1, dict2):
    """
    Helper function to test equality of two data dictionaries

    :param dict_1 (dict {str: Tensor): one data dictionary
    :param dict_2 (dict {str: Tensor): second data dictionary
    :return (bool): True if data dictionaries have same key, and the (value) tensors
       are equal for each key
    """
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


def get_test_dataloader_example_1():
    """ associated dataloaders for example 1 """
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
    """ taken directly from problem.py step to test correctness """
    input_dict = test_data
    for node in problem.nodes:
        output_dict = node(input_dict)
        if isinstance(output_dict, torch.Tensor):
            output_dict = {node.name: output_dict}
        input_dict = {**input_dict, **output_dict}
    return input_dict


def list_equals_modulelist(lst, mod_list):
    """
    Helper function to test if a standard list "equals" a generic iterable (in this case
        a nn.ModuleList)

    :param dict_1 (dict {str: Tensor): one data dictionary
    :param dict_2 (dict {str: Tensor): second data dictionary
    :return (bool): True if data dictionaries have same key, and the (value) tensors
        are equal for each key
    """
    lst2 = []
    for elem in mod_list:
        lst2.append(elem)
    return lst == lst2


def test_problem_initialization():
    """
    Pytest testing function to check initialization of a problem, ensuring its class
    attributes are correct.
    """
    objectives, constraints, components, loss, _ = example_1()
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

    assert list_equals_modulelist(components, problem.nodes)
    assert problem.loss == loss
    assert problem.grad_inference == True
    assert problem.check_overwrite == True
    assert isinstance(problem, torch.nn.Module)

    problem = Problem(components, loss)

    assert list_equals_modulelist(components, problem.nodes)
    assert problem.loss == loss
    assert isinstance(problem.grad_inference, bool)
    assert isinstance(problem.check_overwrite, bool)
    assert isinstance(problem, torch.nn.Module)


def test_problem_graph_generation():
    objectives, constraints, components, loss, expected_edges = example_1()
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

    assert edges == expected_edges


def test_problem_step():
    objectives, constraints, components, loss, edges = example_1()
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)
    test_dataloader = get_test_dataloader_example_1()
    test_data = next(iter(test_dataloader))


    expected_output = step(test_data, problem)
    actual_output = problem.step(test_data)
    assert dict_equals(expected_output, actual_output)


def test_problem_loss():
    objectives, constraints, components, loss, edges = example_1()
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)
    test_dataloader = get_test_dataloader_example_1()
    test_data = next(iter(test_dataloader))
    output = problem(test_data)

    assert 'train_loss' in list(output.keys())

    loss_val = output['train_loss']
    assert loss_val.requires_grad


def test_problem_initialization_faulty():
    objectives, constraints, components, loss, edges = example_1()
    components += ["foo"]
    with pytest.raises(TypeError):
        Problem(components, loss)

    with pytest.raises(TypeError):
        Problem("Foo", loss)

    with pytest.raises(TypeError):
        problem2 = Problem(1, loss)

    with pytest.raises(TypeError):
        problem2 = Problem(nn.Module, loss)


def test_check_keys():
    """
    Pytest testing function to check that private function _check_keys() to check
    uniqueness of node names is correct.
    """
    objectives, constraints, components, loss, edges = example_1()
    node1 = Node(lambda x: x, ['p'], ['p'], name='Node')
    problem2 = Problem(nodes=[node1], loss=loss, grad_inference=True, check_overwrite=True)

    with pytest.warns(Warning):
        problem2._check_keys()



def test_check_unique_names():
    """
    Pytest testing function to check that private function _check_unique_names (within graph()) to check
    uniqueness of node names is correct.
    """
    objectives, constraints, components, loss, edges = example_1()
    new_components = components.copy()

    node2 = Node(lambda x: x, ['p'], ['y'], name='map')
    new_components.append(node2)

    # should not throw an error because names are unique
    problem = Problem(nodes=components, loss=loss, grad_inference=True, check_overwrite=True)

    # should throw an error because node2 has a duplicate name
    with pytest.raises(AssertionError):
        problem2 = Problem(nodes=new_components, loss=loss, grad_inference=True, check_overwrite=True)




