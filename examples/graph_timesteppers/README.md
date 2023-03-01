# Graph Neural Network Timesteppers in Neuromancer
Graph Neural Network Timesteppers offer gray and black box architectures that model pairwise interactions and temporal dynamics. Examples include force dynamics between particles or heat transfer between spaces. Like other Neuromancer models, physical and structural constraints can be imposed on the model dynamics and they can be learned via gradient-based optimization algorithms. 

## Architecture
The graph timestepper class follows an encode-process-decode architecture. The flow of the class is shown below with the inputs and outputs of each module within the timestepper:

![](/examples/figs/gt_architecture.png)

In this architecture, the current state of the system, x0, is embedded into a latent space by the encoder. The processor produces the next state of the system within the latent space and incorporates extrinsic variables (inputs and disturbances) into the prediction. A loopback function allows the processor to use its own output as input to predict states further ahead. Finally the decoder function takes each of these outputs and translates them back into the output dimension from the latent space.

The primary workforce of most Graph Neural Networks and the graph timestepper processor are message passing (MP) operations which can be broken into propagate and update functions. Propagate functions are functions of the state of a node that are passed to each of the neighbors of said node. Information spreads through the graph or network via these operations. The figure below demonstrates the flow of messages in a network including messages from a node to itself:

![](/examples/figs/mp_overview.png)

The second part of message passing is the update step in which the state of each node is updated as a function of its previous state and the set of messages it receives from its neighbors. That is the state $x$ of node $i$ is transformed as

$$
x_i(l+1) = f\left( x_i(l),\ \{g(x_j(l))\ :\ j \in N(i)\} \right)
$$

where $f$ is the update function (e.g. MLP), $g$ is the propagation  function (e.g. linear transform), $l$ is the current layer input and $N(i)$ is the set of neighbors of vertex $i$.

## Graph Timestepper Models

The base graph timestepper class: [GraphTimestepper](#graph-timestepper-models) and the child classes [MLPGraphTimestepper](#mlpgraphtimestepper) and [RCTimestepper](#rctimestepper) are located in [dynamics.py](../../neuromancer/dynamics.py). The base class can be instantiated in the following way:

```python
from neuromancer.dynamics import GraphTimestepper
num_nodes = 4
model = GraphTimestepper(num_nodes)
```

but will not process data without defining its inner modules. Use the child classes or see [implementing custom graph timesteppers](#implementing-custom-graph-timesteppers) for how to construct new models. The class is composed of 4 optional neural network modules that operate in sequence: preprocessor, graph encoder, processor, and decoder. The preprocessor is analogous to including a previous Neuromancer component, the encoder is a one-time call that transforms the data from the preprocessor into the latent space in which the processor operates on it. Output data from the processor can loop back into the processor for multi-step temporal predictions (determined by the number of timesteps in input Yf). Finally the decoder transforms each output of the processor into the output space.

## Optional init parameters:
- preprocessor: Module that acts on input before everything else.
- graph_encoder: Module that transforms input features into the latent dimension.
- processor: Module that processes the graph in the latent space
- decoder: Module to transform from latent space to output space
- fu: Module to process input variables.
- fd: Module to process disturbance variables
- add_batch_norm: Will apply batch normalization to both node and edge features at each step.
- latent_dim: Number of features per node in the processor.
- input_key_map: Dictionary mapping input keys to nondefault names.
- name: Name of the module.
- device: Location to run the module (cpu/cuda)

### Input Keys:
- x0: Tensor of input states with shape: (batch, nodestate). Nodestates should be in the order: node 0 state 0, node 0 state 1, ... , node n state m-1, node n state m.
- Yf: Tensor of ground truth output states with shape: (batch, time, nodestate)
- (opt) Uf: Tensors of external inputs with shape: (batch, time, |U|)
- (opt) Df: Tensor of external disturbances with shape (batch, time, |D|)

### Output Keys:
- X_pred: Predicted X values (output of processor) with shape: (batch, time, latent_dim)
- Y_pred: Predicted Y values (output of decoder) with shape: (batch, time, output_dim)

## MLPGraphTimestepper
The **MLPGraphTimestepper** is a black box model that requires the number of nodes in the graph and the number of node and edge features to instantiate. The preprocessor for the MLPGraphtimestepper default to an identity function unless specified otherwise by the user (e.g. another neuromancer module). The graph encoder and decoder can also be provided as modules (see [implementation](#implementing-custom-graph-timesteppers)]) or will be constructed as MLPs that operate on the node and edge features.

The default graph encoder uses a pair of MLPs to transform the input node features and edge features into the latent dim. The shape of the MLPs is determined by hidden_sizes. The default decoder, similarly, is an MLP that transforms the node_attr produced by the processor from the latent_dim to out_size. 

The processor is composed of paired multi-layer perceptrons (MLP) with hidden layers specified by hidden_sizes, and output shape as latent_dim.

$$
\frac{d}{dt}e_{ij}(t) = MLP(|e_{ij}(t)\ x_i(t)\ x_j(t)|)\\
\frac{d}{dt}x_i(t) = MLP(|x_i(t)\ \sum_{j}e_{ij}(t)|)
$$

such that at each timestep, the change in the edge features is determined by the output of a MLP which takes in the last timestep's edge features concatenated with the incident node features. The change in the node features is the output of a separate MLP of the previous timestep's node features and the sum of all the incident edge features. Unless an integrator is specified, the new features will replace the old features outright.

Below is sample code to construct the model using the NodeDeltaPreprocessor which constructs edge features from the difference between adjacent node features. If no edge_index is specified, it assumes a fully connected graph:

```python
import torch
from neuromancer.dynamics import MLPGraphTimestepper
from neuromancer.gnn import NodeDeltaPreprocessor

#Examples architecture for problem with 8 bodies
features, latent_dim = 6, 8
model = MLPGraphTimestepper(
    num_nodes = 8, #Number of nodes in the graph
    num_node_features = features, #Input features per node
    num_edge_features = features, #Input features per edge
    out_size = 1, #Output size of decoder per node for a total of 8 output states
    latent_dim = latent_dim, #Output size of all MLPs in encoder/decoder
    hidden_sizes = [16, 16], #Used for all constructed MLP
    message_passing_steps = 2, #How far information spreads in the graph each timestep
    preprocessor = NodeDeltaPreprocessor(
        attr_key = 'x0',
        in_features = 6,
        transform = torch.nn.Linear(features, features)),
    integrator = 'Euler', #Integration scheme
    add_batch_norm = True, #Batch normalize each message passing step
    name = 'model_mlp_graph_timestepper', #Component name
    device = 'cpu')

batch_size, prediction_horizon = 1, 4
data = {'x0': torch.rand(batch_size, 8*features), 
         'Yf': torch.rand(batch_size, prediction_horizon, 1)}
output = model(data)
```

### Optional init parameters:
- latent_dim: Number of features per node in the processor.
- hidden_size: List of hidden sizes for each MLP (Includes graph_encoder)
- message_passing_steps: Number of iterations of the MLP in the processor per timestep. Larger numbers will spread information further across the graph.
- preprocessor: Module that acts on input before everything else.
- graph_encoder: Module that transforms input features into the latent dimension.
- integrator: Neuromancer integrator scheme for the processor
- decoder: Module to transform from latent space to output space
- fu: Module to process input variables.
- fd: Module to process disturbance variables
- add_batch_norm: Will optionally batch normalize node and edge features at each step.
- latent_dim: Integer denoting the number of features per node in the processor.
- input_key_map: Dictionary mapping input keys to nondefault names.
- name: Name of the module.
- device: Location to run the module (cpu/cuda)

## RCTimestepper
The **RCTimestepper** is a gray box model. Resistor-capacitor networks (RC Networks) are physical models for both electrical networks and heat-flow. In the latter case, the resistance corresponds to the heat-resistance of walls in a house, and the capacitance to the heat capacity of each room and networks can be formed by physical adjacencies from room to room. Heat flow in RC networks is modeled via the following equation: 

$$
\frac{d}{dt}x_i(t) = \sum_{j \in N(i)}\frac{1}{R_{ij}C_{i}}(x_j(t)-x_i(t))
$$

where N(i) is the neighborhood set of i. This equation forms the basis of the message passing operation in the processor of the **RCTimestepper** with R and C being learnable parameters. 

By default the RCTimestepper constructs an RCPreprocessor (see [gnn.py](/neuromancer/gnn.py)). The preprocessor constructs node_attr from x0, as well as edge_attr from the node_attrr difference between adjacent nodes.

The processor is constructed to follow the above RC equation. The provided integration scheme will be used to integrate the input state using the output of the RC equation. Setting message_passing_steps greater than 1 will compute the RC update multiple sequential times per processor call, updating the node attributes after each one and using the new features for the next call. They will not share R and C parameters.

The graph_encoder and decoder default to identity functions unless specified by the user. See [below](#implementing-custom-graph-timesteppers) for how to add custom encoders and decoders.

Constructing an RCTimestepper only requires an edge_index. The edge_index is a longtensor with shape (2, #edges) where edge_index\[0,i\] corresponds to the head node of the $i$th edge of a graph and edge_index\[1,i\] the tail node. RCTimesteppers require undirected graphs (i.e. if (a,b) is in the edge_index, (b,a) must also be). Below is an implementation of an RCTimestepper:

```python
import torch
from neuromancer.dynamics import RCTimestepper

#edge_index is an undirected version of the graph at the start of the readme
edge_index = torch.tensor([[0,0,0,1,1,2,2,3],
                           [1,2,3,0,2,0,1,0]], dtype=torch.long)
model = RCTimestepper(edge_index)

batch_size, prediction_horizon = 1,4
data = {'x0': torch.rand(batch_size, 4), 'Yf': torch.rand(batch_size, 4)}
output = model(data)
```

A more detailed example of the RCTimestepper can be seen in the [rc_network example](rc_network.py). 

### Optional init parameters
- node_dim: Number of features per node (after preprocessor/encoder)
- nodes: Number of nodes in the graph
- message_passing_steps: Number of iterations of the MLP in the processor per timestep. Larger numbers will spread information further across the graph.
- preprocessor: Module that acts on input before everything else.
- graph_encoder: Module that transforms input features into the latent dimension.
- integrator: Neuromancer integrator scheme for the processor
- decoder: Module to transform from latent space to output space
- fu: Module to process input variables.
- fd: Module to process disturbance variables
- input_key_map: Dictionary mapping input keys to nondefault names.
- name: Name of the module.
- device: Location to run the module (cpu/cuda)

## Implementing Custom Graph Timesteppers
The base graph timestepper class can be used to implement a wide range of desired operations by instantiating it with your own modules. The only required parameter is the num_nodes, the number of nodes in the graph that will be processed. Internally, the timestepper processes information in the form of GraphFeatures (see [gnn.py](../../neuromancer/gnn.py)) which can be treated as dictionaries with the keys "node_attr" and "edge_attr" pointing to tensors of shape (batch, number of nodes or edges, latent_dim). GraphFeatures provides overloaded operations for arithmetic and concatenation. Archetypes for each of the optional graph timestepper module are provided below:

**preprocessor**: If not provided, inputs to the timestepper are expected to include edge_index, node_attr, and edge_attr tensors.
```python
class preprocessor(nn.Module):   
    def forward(data: dict) -> dict:
        """
        :param data: {"x0":Tensor (batch, node/states), "Yf": Tensor (bathc, timesteps, node/states)}
        :return: {"edge_index":Tensor (2, edges), "node_attr":Tensor (batch, nodes, latent_dim), "edge_attr":Tensor (batch, edges, latent_dim)}
        """
```

**graph_encoder**: One time graph encoder, not part of the rollout loop
```python
class preprocessor(nn.Module):   
    def forward(G: GraphFeatures) -> GraphFeatures:
        """
        :param G: {"node_attr":Tensor (batch, nodes, latent_dim), "edge_attr":Tensor (batch, edges, latent_dim)}
        :return: {"node_attr":Tensor, "edge_attr":Tensor}
        """
```

**processor**: ModuleList containing the message passing operations. Called at each timestep
```python
class mp_op(nn.Module):   
    def forward(G: GraphFeatures, u: dict) -> (GraphFeatures):
        """
        :param G: GraphFeatures
        :param u: dict, {"edge_index": Tensor, "edge_attr": Tensor, "node_attr": Tensor}
        :return: GraphFeatures
        """

processor = nn.ModuleList([mp_op() for _ in range(message_passing_steps)])
```

**decoder**: Operates on each output of the processor, note that edge attributes are not processed by the decoder
```python
class decoder(nn.Module):   
    def forward(X: Tensor) -> Tensor:
        """
        :param X: Tensor, Latent States of nodes, shape: (batch, node/states)
        :return: Tensor, Output variable, shape: (batch, node/states)
        """
```

**fu** and **fd**: Treated similarly, example given for fu, for fd the second input variable is D rather than U.
```python
class fu(nn.Module):   
    def forward(G: GraphFeatures, U : Tensor) -> Tensor:
        """
        :param G: Graph Features
        :param U: Tensor, Input Variables for current time, shape: (batch, *)
        :return: Tensor, Output variable, shape: (batch, node/states)
        """
```
