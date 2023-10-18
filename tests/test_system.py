import pytest
import torch
import torch.nn as nn
import pydot
import itertools
from neuromancer.system import Node, System
from collections import defaultdict
from hypothesis import given, settings, strategies as st

torch.manual_seed(0)

"""
################################ Testing Node ########################################
"""

class TestNode:
    """
    Testing class for node
    """
    def setup_method(self):
        # Set up sample data for testing
        self.sample_data = {
            'x1': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'x2': torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        }

        self.sample_callable_tuple_output = lambda x1,x2: (x1 + x2, x1 - x2)
        self.sample_callable_single_output = lambda x1, x2: x1 + x2
        self.sample_callable_overriding = lambda x1,x2: x2


    def test_node_initialization(self):
        # Test the initialization of the Node class

        node = Node(self.sample_callable_tuple_output, ['x1', 'x2'], ['y1', 'y2'], name='test_node')
        assert node.input_keys == ['x1', 'x2']
        assert node.output_keys == ['y1', 'y2']
        assert node.name == 'test_node'
        assert node.callable == self.sample_callable_tuple_output

    def test_node_forward(self):
        # Test the forward method of the Node class on tuple input/output

        node = Node(self.sample_callable_tuple_output, ['x1', 'x2'], ['y1', 'y2'])
        data_dict = self.sample_data
        result = node(data_dict)

        assert 'y1' in result
        assert 'y2' in result
        assert torch.all(result['y1'] == self.sample_data['x1'] + self.sample_data['x2'])
        assert torch.all(result['y2'] == self.sample_data['x1'] - self.sample_data['x2'])

    def test_node_forward_single_output(self):
        # Test the forward method of the Node class on tuple input, scalar output

        node = Node(self.sample_callable_single_output, ['x1', 'x2'], ['y1', 'y2'])
        data_dict = self.sample_data
        result = node(data_dict)

        assert 'y1' in result
        assert 'y2' not in result
        assert torch.all(result['y1'] == self.sample_data['x1'] + self.sample_data['x2'])

    def test_node_forward_missing_keys(self):
        # Test forward when data dictionary has missing keys

        node = Node(self.sample_callable_tuple_output, ['z1'], ['y1', 'y2'])
        data_dict = self.sample_data

        with pytest.raises(KeyError):
            node(data_dict)

    def test_node_forward_extra_keys(self):
        # Test forward when data dict has extra keys

        node = Node(self.sample_callable_tuple_output, ['x1', 'x2', 'z1'], ['y1', 'y2'])
        data_dict = self.sample_data

        with pytest.raises(KeyError):
            node(data_dict)

    def test_node_forward_callable_override(self):
        # test forward when the callable has returns a different number of values than
        # that expected by the node's output keys

        node = Node(self.sample_callable_overriding, ['x1', 'x2'], ['y1', 'y2'])
        data_dict = self.sample_data
        result = node(data_dict)
        keys = list(result.keys())
        assert len(keys) == 1
        assert 'y1' in keys
        assert 'y2' not in keys

    def test_node_same_keys(self):
        # test node whose input and output keys are the same

        node = Node(lambda x,y: (x,y), ['x1', 'x2'], ['x1', 'x2'])
        data_dict = self.sample_data
        result = node(data_dict)
        assert dict_equals(result, data_dict)


"""
################################ Testing System ########################################

TO DO: Encapsulate nodes and their respective adjacency list into a Class

"""


def sample_basic_nodes():
    # Create list of nodes that form a "basic" DAG

    node_1 = Node(callable=lambda x: x, input_keys=['x1'], output_keys=['y1'], name='node_1')

    net_2 = torch.nn.Sequential(torch.nn.Linear(2, 5),
                                torch.nn.ReLU(),
                                torch.nn.Linear(5, 3),
                                torch.nn.ReLU(),
                                torch.nn.Linear(3, 1))
    node_2 = Node(callable=net_2, input_keys=['x2'], output_keys=['y2'], name='node_2')
    node_3 = Node(callable=lambda x1, x2: 2.*x1 - x2**2, input_keys=['y1', 'y2'], output_keys=['y3'], name='quadratic')
    return [node_1, node_2, node_3]


def sample_basic_nodes_without_names():
    # Create list of nodes that form a "basic" DAG, nodes are without names
    # to check if system() properly instantiates them

    node_1 = Node(callable=lambda x: x, input_keys=['x1'], output_keys=['y1'])

    net_2 = torch.nn.Sequential(torch.nn.Linear(2, 5),
                                torch.nn.ReLU(),
                                torch.nn.Linear(5, 3),
                                torch.nn.ReLU(),
                                torch.nn.Linear(3, 1))
    node_2 = Node(callable=net_2, input_keys=['x2'], output_keys=['y2'])
    node_3 = Node(callable=lambda x1, x2: 2.*x1 - x2**2, input_keys=['y1', 'y2'], output_keys=['y3'])
    return [node_1, node_2, node_3]


def sample_basic_nodes_without_names_edges():
    # The edges associated with sample_basic_nodes_without_names

    edges = defaultdict(list,
            {'node_2': ['node_3', 'out'],
             'node_1': ['node_3', 'out'],
             'in': ['node_1', 'node_2'],
             'node_3': ['out']})
    return dict(edges)


def sample_basic_nodes_edges():
    # The edges associated with sample basic nodes

    edges = defaultdict(list,
                       {'node_2': ['quadratic', 'out'],
                        'node_1': ['quadratic', 'out'],
                        'in': ['node_1', 'node_2'],
                        'quadratic': ['out']})
    return dict(edges)


def sample_isolated_graph_nodes():
    # Create list of nodes that form an isolated graph (isolated from dataset)

    node_1 = Node(callable=lambda x: x, input_keys=['x1'], output_keys=['y1'], name='node_1')
    node_2 = Node(callable=lambda x: x, input_keys=['y1'], output_keys=['x1'], name='node_2')
    return [node_1, node_2]


def sample_isolated_graph_nodes_edges():
    # edges associated with isolated graph

    edges = defaultdict(list, {'node_1': ['node_2', 'out'], 'node_2': ['out']})
    return dict(edges)


def sample_single_node_basic():
    # create a node list containing single node

    node_1 = Node(callable=lambda x: x, input_keys=['x1'], output_keys=['y1'], name='node_1')
    return [node_1]


def sample_single_node_basic_edges():
    # edges for the single node graph

    edges = defaultdict(list, {'in': ['node_1'], 'node_1': ['out']})
    return dict(edges)

def sample_single_node_recurrent():
    # create a node list containing a single node with self-loop

    node_1 = Node(callable=lambda x: x, input_keys=['x1'], output_keys=['x1'], name='node_1')
    return [node_1]

def sample_single_node_recurrent_edges():
    # edges for the self-loop single-node graph

    edges = defaultdict(list, {'node_1': ['node_1', 'out'], 'in': ['node_1']})
    return dict(edges)


# Define fixtures for different node/adjacency list pairs
@pytest.fixture(params=[(sample_basic_nodes(), sample_basic_nodes_edges()), \
                        (sample_basic_nodes_without_names(), sample_basic_nodes_without_names_edges()), \
                        (sample_isolated_graph_nodes(), sample_isolated_graph_nodes_edges()), \
                        (sample_single_node_basic(), sample_single_node_basic_edges()), \
                        (sample_single_node_recurrent(), sample_single_node_recurrent_edges() )
                        ])
def get_nodes_and_edges(request):
    return request.param


# Define a fixture for testing range of n_steps and batch_sizes
@pytest.fixture(params=[(0, 0), (1, 1), (1, 2), (2, 2), (2, 50)])
def get_nstep_batch(request):
    return request.param


#sample callable to operate on data dictionaries
#to be used in get_init_func_error_pairs
def h(data_dict):
    for key in data_dict:
        data_dict[key] = data_dict[key] ** 2
    return data_dict


# Fixture to create (init_func, expected_error) pairs
@pytest.fixture(params=[(lambda x: x, None),(lambda x: x+1, TypeError), (h, None)])
def get_init_func_error_pairs(request):
    return request.param


def get_input_value_count(nodes):
    input_value_count = {}
    for node in nodes:
        node_name = node.name
        if isinstance(node.callable, torch.nn.Module):

            first_layer = node.callable[0]
            if hasattr(first_layer, 'in_features'):
                # If the callable has an 'in_features' attribute, it's a linear layer
                input_value_count[node_name] = first_layer.in_features
        else:
            # For other callables, check the number of input keys
            input_value_count[node_name] = len(node.input_keys)
    return input_value_count


def generate_data_dict(sample_nodes, expected_edges, nstep, batch):

    data_dict = {}
    input_value_counts = get_input_value_count(sample_nodes)
    if 'in' in list(expected_edges.keys()):
        input_node_names = expected_edges['in']
    else:
        input_node_names = list(expected_edges.keys())
    input_nodes = [n for n in sample_nodes if n.name in input_node_names]
    input_node_names = [n.name for n in input_nodes]
    input_keys = [n.input_keys for n in input_nodes]
    input_keys = list(itertools.chain(*input_keys))

    idx = 0
    for input_key in input_keys:
        node_name = input_node_names[idx]
        # Generate a random tensor of shape [batch x nstep x 1]
        tensor = torch.rand(batch, nstep, input_value_counts[node_name])
        data_dict[input_key] = tensor
        idx += 1

    return data_dict


def generate_expected_output(node_list, nsteps, init_data):
    expected_data = init_data.copy()
    for i in range(nsteps):
        for node in node_list:
            indata = {k: expected_data[k][:, i] for k in node.input_keys}
            outdata = node(indata)
            expected_data = cat(expected_data, outdata)  # feed the data nodes
    return expected_data


def dict_equals(dict1, dict2):

    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key not in dict2:
            return False
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        if not torch.equal(tensor1, tensor2):
            return False
    return True

def list_equals_modulelist(lst, mod_list):
    lst2 = []
    for elem in mod_list:
        lst2.append(elem)
    return lst == lst2

def cat(data3d, data2d):
    """
    Concatenates data2d contents to corresponding entries in data3d
    :param data3d: (dict {str: Tensor}) Input to a node
    :param data2d: (dict {str: Tensor}) Output of a node
    :return: (dict: {str: Tensor})
    """
    for k in data2d:
        if k not in data3d:
            data3d[k] = data2d[k][:, None, :]
        else:
            data3d[k] = torch.cat([data3d[k], data2d[k][:, None, :]], dim=1)
    return data3d


def is_valid_node_list(nodes):
    # Ensure all child nodes are to the right of parent nodes
    # TO DO: CHECK CORRECTNESS on SINGLE NODE CASE

    dependency_dict = dict()
    for node in nodes:
        output_keys, in_keys = node.output_keys, node.input_keys
        for o in output_keys:
            if o not in dependency_dict:
                dependency_dict[o] = in_keys
            else:
                dependency_dict[o].append(in_keys)

    visited = set()
    for node in nodes:
        if not any(i in list(dependency_dict.keys()) for j in range(len(node.output_keys)) for i in
                   dependency_dict[node.output_keys[j]]):
            for n in node.output_keys:
                visited.add(n)
        else:
            for n in node.input_keys:
                if not n in visited:
                    return False
    return True


def test_system_initialization(get_nodes_and_edges, get_nstep_batch):
    sample_nodes, expected_edges = get_nodes_and_edges

    test_init_func = None
    test_nstep_key = 'X'
    test_nsteps = 3

    system = System(nodes=sample_nodes, nstep_key=test_nstep_key, init_func=test_init_func, nsteps=test_nsteps)
    assert system is not None
    assert isinstance(system.nodes, torch.nn.ModuleList)
    assert list_equals_modulelist(sample_nodes, system.nodes)
    assert hasattr(system.init, '__self__') #original init
    assert system.nstep_key == test_nstep_key
    assert system.nsteps == test_nsteps




def test_graph_generation_valid_node_lists(get_nodes_and_edges):
    sample_nodes, expected_edges = get_nodes_and_edges
    system = System(nodes=sample_nodes)
    graph = system.system_graph
    assert graph is not None
    assert isinstance(graph, pydot.Dot)

    # Correct node names if they are unnamed
    input_keys = []
    output_keys = []
    nonames = 1
    for node in sample_nodes:
        input_keys += node.input_keys
        output_keys += node.output_keys
        if node.name is None or node.name == '':
            node.name = f'node_{nonames}'
            nonames += 1

    expected_node_names = [node.name for node in sample_nodes]  # expected_node_names
    for node in system.nodes:  # node name after being ingested by system
        assert node.name in expected_node_names

    # Edge Testing -- determine if edges correctly instantiated
    edge_list = system.system_graph.get_edges()
    edges = defaultdict(list)
    for e in edge_list:
        src, dest = e.get_source(), e.get_destination()
        edges[src].append(dest)

    assert edges == expected_edges

"""
def test_graph_generation_invalid_node_lists(get_nodes_and_edges):
    sample_nodes, expected_edges = get_nodes_and_edges
    system = System(nodes=sample_nodes)
    graph = system.system_graph
    assert graph is not None
    assert isinstance(graph, pydot.Dot)

    # Correct node names if they are unnamed
    input_keys = []
    output_keys = []
    nonames = 1
    for node in sample_nodes:
        input_keys += node.input_keys
        output_keys += node.output_keys
        if node.name is None or node.name == '':
            node.name = f'node_{nonames}'
            nonames += 1

    expected_node_names = [node.name for node in sample_nodes]  # expected_node_names
    for node in system.nodes:  # node name after being ingested by system
        assert node.name in expected_node_names

    # Edge Testing -- determine if edges correctly instantiated
    edge_list = system.system_graph.get_edges()
    edges = defaultdict(list)
    for e in edge_list:
        src, dest = e.get_source(), e.get_destination()
        edges[src].append(dest)

    assert edges == expected_edges
"""


def test_system_init(get_nodes_and_edges, get_nstep_batch, get_init_func_error_pairs):
    sample_nodes, expected_edges = get_nodes_and_edges
    nstep, batch = get_nstep_batch
    init_func, expected_error = get_init_func_error_pairs

    input_data_dict = generate_data_dict(sample_nodes, expected_edges, nstep, batch)
    system = System(nodes=sample_nodes, nsteps=nstep, init_func=init_func)

    if expected_error is not None:
        with pytest.raises(expected_error):
            output_data_dict = system.init(input_data_dict)
    else:
        expected_data_dict = init_func(input_data_dict)
        output_data_dict = system.init(input_data_dict)
        assert dict_equals(expected_data_dict, output_data_dict)


def test_system_cat():
    callable = lambda x: x*2
    nsteps = 3
    batch_size = 2
    node_1 = Node(callable, ['x1'], ['y1'])
    system = System(nodes=[node_1],nsteps=3)
    input_data_dict = {'x1': torch.rand(batch_size, nsteps, 1)}
    output_data_dict = node_1(input_data_dict)

    test_cat_result = system.cat(input_data_dict, output_data_dict)
    expected_cat_result = cat(input_data_dict, output_data_dict)

    assert dict_equals(test_cat_result, expected_cat_result)




def test_forward_on_valid_node_lists(get_nodes_and_edges, get_nstep_batch):
    sample_nodes, expected_edges = get_nodes_and_edges
    nstep, batch = get_nstep_batch
    system = System(nodes=sample_nodes, nsteps=nstep)
    input_data_dict = generate_data_dict(sample_nodes, expected_edges, nstep, batch)
    test_result_dict = system(input_data_dict)
    expected_result_dict = generate_expected_output(node_list=sample_nodes, nsteps=nstep, init_data=input_data_dict)
    assert dict_equals(test_result_dict, expected_result_dict)

"""
def test_forward_on_invalid_node_lists(get_nodes_and_edges, get_nstep_batch):

    sample_nodes, expected_edges = get_nodes_and_edges

    nstep, batch = get_nstep_batch
    if 'in' in list(expected_edges.keys()):
        nodes_list = list(itertools.permutations(sample_nodes))
        invalid_nodes_list = []
        for lst in nodes_list:
            if not is_valid_node_list(lst):
                invalid_nodes_list.append(lst)

        for nodes in invalid_nodes_list:
            if len(nodes) > 1: #only applies to graphs with > 1 node
                nodes = list(nodes)
                system = System(nodes=nodes, nsteps=nstep)
                input_data_dict = generate_data_dict(nodes, expected_edges, nstep, batch)
                print(nodes)
                print(system(input_data_dict))

                with pytest.raises(KeyError):
                    _ = system(input_data_dict)

"""