"""
NeuroMANCER Node and System classes and modules tutorial

This script demonstrates how to use NeuroMANCER Node to wrap arbitrary callable
into symbolic representation that can be used in NeuroMANCER problem formulation.

"""

import torch
from neuromancer.system import Node, System

"""
Node is a simple class to create symbolic modules out of arbitrary PyTorch callables.
Node class is wrapping the callable and defines the computational node based 
on input_keys and output_keys that define computational node connections through 
intermediate dictionaries. Complex symbolic architectures can be constructed by connecting
input and output keys of a set of Nodes via System and Problem classes.
    
"""

# 1, wrap nn.Linear into Node
net_1 = torch.nn.Linear(1, 1)
node_1 = Node(net_1, ['x1'], ['y1'])
# print input and output keys
print(node_1.input_keys)
print(node_1.output_keys)
# evaluate forward pass of the node with dictionary input dataset
print(node_1({'x1': torch.rand(1)}))

# 2, wrap nn.Sequential into Node
net_2 = torch.nn.Sequential(torch.nn.Linear(2, 5),
                            torch.nn.ReLU(),
                            torch.nn.Linear(5, 3),
                            torch.nn.ReLU(),
                            torch.nn.Linear(3, 1))
node_2 = Node(net_2, ['x2'], ['y2'])
# evaluate forward pass of the node with dictionary input dataset
print(node_2({'x2': torch.rand(2)}))

# 3, wrap arbitrary callable into Node - allows for unwrapping the inputs
fun_1 = lambda x1, x2: 2.*x1 - x2**2
node_3 = Node(fun_1, ['y1', 'y2'], ['y3'], name='quadratic')
# evaluate forward pass of the node with dictionary input dataset
print(node_3({'y1': torch.rand(2), 'y2': torch.rand(2)}))

# 4, wrap callable with multiple inputs and outputs
def fun_2(x1, x2):
    return x1**2, x2**2
node_4 = Node(fun_2, ['x1', 'x2'], ['x1^2', 'x2^2'], name='square')
# evaluate forward pass of the node with dictionary input dataset
print(node_4({'x1': torch.rand(2), 'x2': torch.rand(2)}))


"""
NeuroMANCER also provides implementation of useful building blocks for
creating custom neural architectures. These include:
    modules.blocks          - neural architecures
    modules.activations     - custom activation functions    
    modules.functions       - useful callables 
    modules.gnn             - graph neural nets
    modules.rnn             - recurent neural nets
    modules.solvers         - iterative solvers for constrained optimization
    slim.linear             - linear algebra factorizations for weights
        
Next set of example shows how to wrap NeuroMANCER modules into Node
"""

from neuromancer.modules import blocks
from neuromancer.modules import activations
from neuromancer import slim

# 1, instantiate 4-layer multilayer perceptron with linear weight and ReLU activation
block_1 = blocks.MLP(insize=2, outsize=3,
                  bias=True,
                  linear_map=slim.maps['linear'],
                  nonlin=torch.nn.ReLU,
                  hsizes=[80] * 4)
# wrap modules into Node
node_4 = Node(block_1, ['x3'], ['y3'])
# evaluate forward pass of the node with dictionary input dataset
data = {'x3': torch.rand(10, 2)}
print(node_4(data).keys())
print(node_4(data)['y3'].shape)

# 2, instantiate recurrent neural net without bias, SVD linear map, and BLI activation
block_2 = blocks.RNN(insize=2, outsize=2,
                  bias=False,
                  linear_map=slim.linear.SVDLinear,
                  nonlin=activations.BLU,
                  hsizes=[80] * 4)
# wrap modules into Node
node_5 = Node(block_2, ['x4'], ['y4'])
# evaluate forward pass of the node with dictionary input dataset
data = {'x4': torch.rand(10, 2)}
print(node_5(data).keys())
print(node_5(data)['y4'].shape)

# for a full list of available blocks (nn.Modules) in NeuroMANCER see:
print(blocks.blocks)
# for a full list of available activations in NeuroMANCER see:
print(activations.activations)

"""
System is a class that supports construction of cyclic computational graphs in NeuroMANCER.
System's graph is defined by a list of Nodes. Instantiated System can be used to simulate
dynamical systems in open or closed loop rollouts by specifying number of steps via nsteps.

"""

# 1, create acyclic symbolic graph
# list of nodes to construct the graph
nodes = [node_1, node_2, node_3]
# 10 steps rollout
nsteps = 3
# connecting nodes via System class
system_1 = System(nodes, nsteps=nsteps)
# print input and output keys
print(system_1.input_keys)
print(system_1.output_keys)
# evaluate forward pass of the System with 3D input dataset
batch = 2
print(system_1({'x1': torch.rand(batch, nsteps, 1),
                'x2': torch.rand(batch, nsteps, 2)}))
# visualize symbolic computational graph
# system_1.show()

# 2, close the loop by creating recursion in one of the nodes
nodes[2].output_keys = ['y1']
# create new system with cyclic computational graph
system_2 = System(nodes, nsteps=nsteps)
# print input and output keys
print(system_2.input_keys)
print(system_2.output_keys)
# evaluate forward pass of the System with 3D input dataset
print(system_1({'x1': torch.rand(batch, nsteps, 1),
                'x2': torch.rand(batch, nsteps, 2)}))
# visualize symbolic computational graph
# system_1.show()
