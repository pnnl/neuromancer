import torch
import torch.nn.functional as F
from neuromancer.estimators import MLPEstimator
from neuromancer.dynamics import block_model
from neuromancer.problem import Problem
from neuromancer import blocks
from neuromancer.activations import activations
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable

import neuromancer.slim as slim


estim = MLPEstimator(
    {"Yp": (10,), "x0": (20,)},
    nsteps=2,
    name="estim"
)

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

data = {
    "Yp": torch.rand(2, 10, 10),
    "Yf": torch.rand(2, 10, 10),
    "Uf_renamed": torch.rand(2, 10, 2),
}
output = {**data, **estim(data)}
output.keys()
output = {**output, **dynamics(output)}
output.keys()
ypred = variable('Y_pred_dynamics')
ytrue = variable('Yf')
xpred = variable('X_pred_dynamics')
x0 = variable('x0_estim')

reference_loss = F.mse_loss(ypred, ytrue).minimize()
bounds_constraint = ypred > 0.5

objectives = [reference_loss]
constraints = [bounds_constraint]
trainable_components = [estim, dynamics]
model = Problem(trainable_components, PenaltyLoss(objectives, constraints))
data["name"] = "test"
output = model(data)
output.keys()