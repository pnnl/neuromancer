# Graph Neural Network Timesteppers in Neuromancer

Graph Neural Network Timesteppers offer gray and black box models that model pairwise interactions in temporal dynamics. Examples include force dynamics between particles or heat transfer between spaces. Like other neuromancer models, physical and structural constraints can be imposed on the model dynamics and they can be learned via gradient-based optimization algorithms. 


### Graph Timestepper Models

The base graph timestepper class: **GraphTimestepper** and the child classes **MLPTGraphTimestepper** and **RCTimestepper** are located in dynamics.py. The base class will not process data without defining its inner modules (use the child classes for most operations) but can be instantiated in the following way:

```python
from neuromancer.dynamics import GraphTimestepper
model = GraphTimestepper(num_nodes)
```

The class is composed of 4 optional neural network modules that operate in sequence: preprocessor, encoder, processor, and decoder. The preprocessor is equivalent to including a previous component, the encoder is a one-time call that transforms the data from the preprocessor into the latent space in which the processor operates on it. Output data from the processor can loop back into the processor for multi-step temporal predictions. Finally the decoder transforms each output of the processor into the output space.

### Optional init parameters:
- preprocessor: Module that acts on input before everything else.
- graph_encoder: Module that transforms input features into the latent dimension.
- processor: Module that processes the graph in the latent space
- decoder: Module to transform from latent space to output space
- fu: Module to process input variables.
- fd: Module to process disturbance variables
- add_batch_norm: Will optionally batch normalize node and edge features at each step.
- latent_dim: Number of features per node in the processor.
- input_key_map: Dictionary mapping input keys to nondefault names.
- name: Name of the module.
- device: Location to run the module (cpu/cuda)

## Input Keys:
- x0: Tensor of input states with shape: (batch, nodestate). Nodestates should be in the order: node 0 state 0, node 0 state 1, ... , node n state m-1, node n state m.
- Yf: Tensor of ground truth output states with shape: (batch, time, nodestate)
- (opt) Uf: Tensors of external inputs with shape: (batch, time, |U|)
- (opt) Df: Tensor of external disturbances with shape (batch, time, |D|)

## Output Keys:
- X_pred: Predicted X values with shape: (batch, time, latent_dim)
- Y_pred: Predicted Y values with shape: (batch, time, output_dim)

## MLPGraphTimestepper
The **MLPGraphTimestepper** is a black box model that is instantiated with the following mandatory arguments:

```python
from neuromancer.dynamics import MLPGraphTimestepper
model = MLPGraphTimestepper(num_nodes, num_features, num_edge_features, out_size)
```
The MLPGraphTimestepper uses a default encoder and decoder that transform node and edge features to and from the latent space using multi-layer perceptrons (MLP). The processor for this timestepper follows these equations:

$$
\frac{d}{dt}e_{ij}(t) = MLP(|e_ij(t) x_i(t) x_j(t)|)\\
\frac{d}{dt}x_i(t) = MLP(|x_i(t) \sum_{j}e_{ij}(t)|)
$$

such that at each timestep, the change in the edge features is determined by the output of an MLP of the last timestep's edge features concatenated with the incident node features. The change in the node features is the output of a separate MLP of the previous timestep's node features and the sum of all the incident edge features.

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
The **RCTimestepper** is a gray box model that is instantiated as follows:

```python
from neuromancer.dynamics import RCTimestepper
model = RCTimestepper(edge_index)
```

Resistor-capacitor networks (RC Networks) are physical models for both electrical networks and heat-flow. For example in the latter case, the resistance corresponds to the heat-resistance of walls in a house, and the capacitance to the heat capacity of each room. Networks are formed by physical adjacencies of one room to the next. Heat flow in RC networks is modeled via the following equation: 

$$
\frac{d}{dt}x_i(t) = \sum_{j \in N(i)}\frac{1}{R_{ij}C_{i}}(x_j(t)-x_i(t)) \\
$$
where N(i) is the neighborhood set of i. This equation forms the basis of the processor in the **RCTimestepper** with R and C being learnable parameters. 

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
