"""
Component tutorial

This script demonstrates how components work together in NeuroMANCER,
and how data flows through the computational graph assembled by the `Problem` class.
We'll demonstrate by building a neural state space model for partially-observable dynamical systems.

"""

import torch
import torch.nn.functional as F

from neuromancer.estimators import MLPEstimator
from neuromancer.dynamics import block_model, BlockSSM
from neuromancer.problem import Problem
from neuromancer.constraint import Loss, variable
from neuromancer.blocks import MLP
from neuromancer.loss import PenaltyLoss
import neuromancer.slim as slim

"""
We begin constructing our neural SSM by creating a latent state estimator to predict initial conditions 
from past observations. To do this, we specify the dimensionality of both our observables and the latent state; 
in this case, we choose 10 and 20, respectively.

As you will see after running the cell below, each component has a handy string representation that indicates 
the name of the component, its input variables, and its outputs. Notice that the outputs of the component 
are tagged with the name of the component that produced them; this is used to prevent name collisions 
in the computational graph, and will become important in the next step.
"""

# instantiating pre-refined state estimator component
estim = MLPEstimator(
    {"Yp": (10,), "x0": (20,)},
    nsteps=2,
    name="estim"
)
# the estimator component is a mapping: estim(Yp) -> x0_estim, reg_error_estim
print(estim)

"""
Next, we define our state space model. The SSM component will take the output of the estimator 
as its initial condition `x0`. However, recall that components tag their outputs with their name. 
If we look at the default input keys of the `BlockSSM` class, we'll notice a slight mismatch:
"""
print(BlockSSM.DEFAULT_INPUT_KEYS)

"""
The canonical name for the initial state variable in `BlockSSM`s is named `x0`, not `x0_estim`. 
Because of this, we need to remap the estimator's output `x0_estim` to `x0`. To do this, 
we hand a dictionary to the `input_keys` parameter of the `block_model` function which maps 
the tagged variable to the canonical name used by the dynamics model. The following cell uses 
the named constructor `block_model` to generate a Block Nonlinear neural SSM:
"""

# instantiating pre-refined neural state space model (SSM) component via helper function block_model()
dynamics = block_model(
    "blocknlin",
    {"x0": (20,), "Yf": (10,), "Uf": (2,)},
    slim.Linear,
    MLP,
    bias=False,
    input_key_map={"x0": "x0_estim", "Uf": "Uf_renamed"},
    name="dynamics"
)
# mapping: dynamics(Yf, x0_estim, Uf_renamed) -> X_pred_dynamics, Y_pred_dynamics, reg_error_dynamics
print(dynamics)

"""
Now we have both model components ready to compose with a `Problem` instance. 
However, before we do that, let's pick apart how data flows through the computational graph 
formed by these components when composed in a `Problem`.

First, we'll create a `DataDict` containing the constant inputs required by each component 
(`Yp`, `Yf`, and `Uf` for the dynamics model; and `Yp` for the estimator).
"""
data = {
    "Yp": torch.rand(2, 10, 10),
    "Yf": torch.rand(2, 10, 10),
    "Uf_renamed": torch.rand(2, 10, 2),
}

"""
Next, let's push the data through the estimator to see what we receive as output 
(note that we combine the data and estimator output to retain the constant inputs 
used by the dynamics model):
"""
output = {**data, **estim(data)}
print(output.keys())

"""
As expected, we obtain our estimated initial state `x0_estim` alongside a `reg_error_estim` 
term measuring the regularization error incurred by any structured linear maps in the component 
(in this case there are none).
Now let's take the output of the estimator and push it through the SSM:
"""
output = {**output, **dynamics(output)}
print(output.keys())

"""
As we can see, the dynamics model correctly handles the `x0_estim` variable; internally, the variable is 
automatically renamed to its canonical name. This capability allows users to combine components in arbitrary ways.

We'll next create some objectives and constraints to demonstrate how these interact with components 
in a `Problem` instance; we define the inputs to each objective and constraint by providing a list of keys 
which can reference either the input data or the output of any component in the overall model.

For alternative way of defining objectives and constraints via high-level variable abstractions see constraints_tutorial.py
"""
reference_loss = Loss(["Y_pred_dynamics", "Yf"], F.mse_loss, name="reference_loss")
estimator_loss = Loss(["X_pred_dynamics", "x0_estim"], lambda x, y: F.mse_loss(x[:-1, -1, :], y[1:]), name="estimator_loss")
bounds_constraint = Loss(["Y_pred_dynamics"], lambda x: F.relu(0.5 - x).mean(), name="bounds_constraint")
ypred = variable('Y_pred_dynamics')
bounds_constraint = ypred < 0.5

"""
At last, let's put together a `Problem` class to combine everything. Like `Component`s, when we instantiate a `Problem` 
we can inspect its string representation to get an overview of all the constructs in the model and see how they are put together.
"""
objectives = [reference_loss, estimator_loss]
constraints = [bounds_constraint]
loss = PenaltyLoss(objectives, constraints)

trainable_components = [estim, dynamics]
model = Problem(trainable_components, loss)
print(model)

# plot computational graph
model.plot_graph()
"""
With our `Problem` created, we can now push the data dictionary we previously defined through it to receive the outputs 
of each component and the values of each objective and constraint we specified. Note that we wrap the data into a `DataDict` 
and add a `name` attribute; like the attribute used in `Component`s, this is used to prevent name collisions between variables 
generated by the use of different data splits during training and validation.
"""

data["name"] = "test"
output = model(data)
print(output.keys())