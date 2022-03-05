# %%
import torch
import torch.nn.functional as F
from neuromancer.estimators import MLPEstimator
from neuromancer.dynamics import block_model, BlockSSM
from neuromancer.problem import Problem
from neuromancer.constraint import Loss
from neuromancer import blocks
from neuromancer.plot import plot_model_graph
from neuromancer.activations import activations

import slim
import neuromancer
# %%

# This estimator expands the observation's dimension from 10 to 20. Two layer MLP.

estim = MLPEstimator(
    {"Yp": (10,), "x0": (20,)},
    nsteps=2,
    name="estim"
)

estim

# %%

BlockSSM.DEFAULT_INPUT_KEYS
# %%
dynamics = block_model(
    "blocknlin",
    {"x0": (20,), "Yf": (10,), "Uf": (2,)},
    slim.Linear,
    blocks.MLP,
    bias=False,
    input_key_map={"x0": "x0_estim", "Uf": "Uf_renamed"},
    name="dynamics"
)



fx = blocks.RNN(20, 20, linear_map=slim.maps['linear'],
                nonlin=activations['softexp'], hsizes=[20, 20])

# %%
data = {
    "Yp": torch.rand(2, 10, 10),
    "Yf": torch.rand(2, 10, 10),
    "Uf_renamed": torch.rand(2, 10, 2),
}
# %%
output = {**data, **estim(data)}
output.keys()
# %%
output = {**output, **dynamics(output)}
output.keys()
# %%
reference_loss = Loss(["Y_pred_dynamics", "Yf"], F.mse_loss, name="reference_loss")
estimator_loss = Loss(["X_pred_dynamics", "x0_estim"], lambda x, y: F.mse_loss(x[-1, :-1, :], y[1:]), name="estimator_loss")
bounds_constraint = Loss(["Y_pred_dynamics"], lambda x: F.relu(0.5 - x).mean(), name="bounds_constraint")
# %%
objectives = [reference_loss, estimator_loss]
constraints = [bounds_constraint]
trainable_components = [estim, dynamics]
model = Problem(objectives, constraints, trainable_components)
model
# %%
data["name"] = "test"
output = model(data)
output.keys()

# %%
