# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, jacrev
import copy

from neuromancer import Node
import neuromancer.psl as psl
from neuromancer.system import System, SystemPreview
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase

from eclipse_nn.LipConstEstimator import LipConstEstimator


# ground truth system model
gt_model = psl.nonautonomous.TwoTank()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = 0
umax = 1.
xmin = 0
xmax = 1.


nsteps = 30  # prediction horizon
n_samples = 2000    # number of sampled scenarios

#  sampled references for training the policy
list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
ref = torch.cat(list_refs)
batched_ref = ref.reshape([n_samples, nsteps+1, nref])
# Training dataset
train_data = DictDataset({'x': torch.rand(n_samples, 1, nx),   # sampled initial conditions of states
                          'r': batched_ref}, name='train')

# sampled references for development set
list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
ref = torch.cat(list_refs)
batched_ref = ref.reshape([n_samples, nsteps+1, nref])
# Development dataset
dev_data = DictDataset({'x': torch.rand(n_samples, 1, nx),    # sampled initial conditions of states
                        'r': batched_ref}, name='dev')

# torch dataloaders
batch_size = 200
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                         collate_fn=dev_data.collate_fn,
                                         shuffle=False)


# white-box ODE model with no-plant model mismatch
two_tank_ode = ode.TwoTankParam()                   # ODE system equations implemented in PyTorch
two_tank_ode.c1 = nn.Parameter(torch.tensor(gt_model.c1), requires_grad=False)
two_tank_ode.c2 = nn.Parameter(torch.tensor(gt_model.c2), requires_grad=False)

# integrate continuous time ODE
integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))   # using 4th order runge kutta integrator

# symbolic system model
model = Node(integrator, ['x', 'u'], ['x'], name='model')


# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                    nonlin=activations['elu'], min=umin, max=umax)
policy = Node(net, ['x', 'r'], ['u'], name='policy')

# closed-loop system model
cl_system = System([policy, model], nsteps=nsteps)
# cl_system.show()


# neural net control policy with reference preview
net_preview = blocks.MLP_bounds(insize=nx + (nref*(nsteps+1)), outsize=nu, hsizes=[64, 32],
                    nonlin=activations['elu'], min=umin, max=umax)
policy_with_preview = Node(net_preview, ['x', 'r'], ['u'], name='policy_with_preview')

cl_system_preview = SystemPreview([policy_with_preview, model], name='cl_system_preview',
                nsteps=nsteps, preview_keys_map={'r': ['policy_with_preview']}, # reference preview for neural control policy node
                preview_length={'r': nsteps}, pad_mode='replicate') # replicate last sample in the sequence



# variables
x = variable('x')
ref = variable("r")
jcl = variable("Jcl_norm")
j = variable("J_norm")


# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion


# constraints
state_lower_bound_penalty = 10.*(x > xmin)
state_upper_bound_penalty = 10.*(x < xmax)
terminal_lower_bound_penalty = 10.*(x[:, [-1], :] > ref-0.01)
terminal_upper_bound_penalty = 10.*(x[:, [-1], :] < ref+0.01)
# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'
terminal_lower_bound_penalty.name = 'y_N_min'
terminal_upper_bound_penalty.name = 'y_N_max'


constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
    terminal_lower_bound_penalty,
    terminal_upper_bound_penalty,
]


# list of constraints and objectives
objectives = [regulation_loss]

# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)

# construct constrained optimization problem without reference preview
# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_{k+1})
problem = Problem([cl_system], loss)
# problem.show()


# construct constrained optimization problem with reference preview
# data (x_k, [r_k, ..., r_{k+N}]) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_{k+1})
problem_with_preview = Problem([cl_system_preview], loss)


optimizer = torch.optim.AdamW(problem.parameters(), lr=0.01)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader, dev_loader,
    optimizer=optimizer,
    epochs=200,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50, 
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)

optimizer_with_preview = torch.optim.Adam(problem_with_preview.parameters(), lr=0.01, weight_decay=0.002)
trainer_with_preview = Trainer(
        problem_with_preview,
        train_loader, dev_loader,
        optimizer=optimizer_with_preview,
        epochs=150,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=5,
        patience=50
    )

# Train control policy with reference preview 
best_model_preview = trainer_with_preview.train()
# Load best model with reference preview
trainer_with_preview.model.load_state_dict(best_model_preview)



# L_rob helper function
def single_sample_policy(policy, x_single, r_single):
    # x_single: [nx], r_single: [nr] or  [(N+1), nr]
    r_single = r_single.reshape(-1)  # works for [nr] and [(N+1), nr]
    out = policy({
        'x': x_single.unsqueeze(0),
        'r': r_single.unsqueeze(0),
    })
    return out['u'].squeeze(0)  # [nu]


# Jacobian of policy w.r.t. x (argnums=1 because inputs are (policy, x, r))
jac_pi_wrt_x = jacrev(single_sample_policy, argnums=1)


def jac_norm(policy, x, r):
    # x: [Batch, nx], r: [Batch, nr] or  [(N+1), nr]
    r = r.reshape(r.shape[0], -1)  # works for [B,nr] and [B,(N+1),nr]
    J_pi = vmap(jac_pi_wrt_x, in_dims=(None, 0, 0))(policy, x, r)  # [Batch, nu, nx]
    j = torch.linalg.norm(J_pi, ord='fro', dim=(-2, -1))                   # [Batch]
    return j.unsqueeze(-1)                                                 # [Batch, 1]



class JacNormCallable:
    def __init__(self, policy):
        self.policy = policy
    def __call__(self, x, r):
        return jac_norm(self.policy, x, r)
    
    
## The problem without reference preview + regularization
# warm-start from the trained baseline policy
policy_regNN = copy.deepcopy(policy)

# penalty node
jac_node = Node(JacNormCallable(policy_regNN), ['x', 'r'], ['J_norm'], name='jac')

# closed-loop system that penalizes controller Jacobian
cl_system_regNN = System([policy_regNN, model, jac_node], nsteps=nsteps)

# penalize controller Jacobian
jac_loss = 0.1 * (j == 0.)^2 

objectives_regNN = [regulation_loss, jac_loss]
loss_regNN = PenaltyLoss(objectives_regNN, constraints)

problem_regNN = Problem([cl_system_regNN], loss_regNN)


## The problem with reference preview + regularization
# warm-start from the trained baseline policy
policy_pre_regNN = copy.deepcopy(policy_with_preview)

# penalty node
jac_node = Node(JacNormCallable(policy_pre_regNN), ['x', 'r'], ['J_norm'], name='jac')

# closed-loop system that penalizes controller Jacobian
cl_system_pre_regNN = SystemPreview(
    [policy_pre_regNN, model, jac_node],
    nsteps=nsteps,
    preview_keys_map={'r': [policy_pre_regNN.name, jac_node.name]},
    preview_length={'r': nsteps},
    pad_mode='replicate'
)

# penalize controller Jacobian
jac_loss = 0.1 * (j == 0.)^2 


objectives_pre_regNN = [regulation_loss, jac_loss]
loss_pre_regNN = PenaltyLoss(objectives_pre_regNN, constraints)

problem_pre_regNN = Problem([cl_system_pre_regNN], loss_pre_regNN)


optimizer_regNN = torch.optim.AdamW(problem_regNN.parameters(), lr=0.01)
trainer_regNN = Trainer(
    problem_regNN,
    train_loader, dev_loader,
    optimizer=optimizer_regNN,
    epochs=100,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50, 
)

# Train control policy
best_model_regNN = trainer_regNN.train()
# load best trained model
trainer_regNN.model.load_state_dict(best_model_regNN)


optimizer_pre_regNN = torch.optim.AdamW(problem_pre_regNN.parameters(), lr=0.01)
trainer_pre_regNN = Trainer(
    problem_pre_regNN,
    train_loader, dev_loader,
    optimizer=optimizer_pre_regNN,
    epochs=100,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50, 
)
# Train control policy
best_model_pre_regNN = trainer_pre_regNN.train()
# load best trained model
trainer_pre_regNN.model.load_state_dict(best_model_pre_regNN)


# Creating the test data
nsteps = 1000
step_length = 250
# generate reference
np_refs = psl.signals.step(nsteps + 1, 1, min=xmin, max=xmax, randsteps=4, rng=np.random.default_rng(20))
R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
torch_ref = torch.cat([R, R], dim=-1)
# generate initial data for closed loop simulation
data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
        'r': torch_ref}

# constraints bounds
Umin = umin * np.ones([nsteps, nu])
Umax = umax * np.ones([nsteps, nu])
Xmin = xmin * np.ones([nsteps+1, nx])
Xmax = xmax * np.ones([nsteps+1, nx])

cl_system.nsteps = nsteps
# perform closed-loop simulation
trajectories = cl_system(data)
# plot closed loop trajectories
pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
      R=trajectories['r'].detach().reshape(nsteps + 1, nref),
      U=trajectories['u'].detach().reshape(nsteps, nu),
      Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax,
      figname='cl.png')
# plot phase portrait
pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx),
         figname='phase.png')

# Applying curriculum learning
cl_system_regNN.nsteps = nsteps
# perform closed-loop simulation
trajectories = cl_system_regNN(data)
# plot closed loop trajectories
pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
      R=trajectories['r'].detach().reshape(nsteps + 1, nref),
      U=trajectories['u'].detach().reshape(nsteps, nu),
      Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax,
      figname='cl.png')
# plot phase portrait
pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx),
         figname='phase.png')


# closed-loop simulation with reference preview
cl_system_preview.nsteps = nsteps
trajectories_with_preview = cl_system_preview(data)
pltCL(Y=trajectories_with_preview['x'].detach().reshape(nsteps + 1, nx),
        R=trajectories_with_preview['r'].detach().reshape(nsteps + 1, nref),
        U=trajectories_with_preview['u'].detach().reshape(nsteps, nu),
        Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
pltPhase(X=trajectories_with_preview['x'].detach().reshape(nsteps + 1, nx))


# closed-loop simulation with reference preview
cl_system_pre_regNN.nsteps = nsteps
trajectories_pre_regNN = cl_system_pre_regNN(data)
pltCL(Y=trajectories_pre_regNN['x'].detach().reshape(nsteps + 1, nx),
        R=trajectories_pre_regNN['r'].detach().reshape(nsteps + 1, nref),
        U=trajectories_pre_regNN['u'].detach().reshape(nsteps, nu),
        Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
pltPhase(X=trajectories_pre_regNN['x'].detach().reshape(nsteps + 1, nx))


# helper function for extracting weights from the trained NNs
import re

def extract_eclipse_tensors(model_dict, device='cpu'):
    """
    Extracts weights from a Neuromancer state_dict as a list of Double-Precision Tensors
    suitable for eclipse-nn input.
    
    Args:
        model_dict (dict): The state_dict from best_model (e.g., best_model_reg)
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        weights (list of torch.Tensor): [W1, W2, ..., Wn] in float64
    """
    layer_weights = {}
    
    # Regex to capture layer index from keys like 'nodes.0...linear.0.weight'
    pattern = re.compile(r"linear\.(\d+)\.(weight)")
    
    for key, val in model_dict.items():
        match = pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            
            # CRITICAL FIX: Add .double() to match eclipse-nn's float64 requirement
            w_tensor = val.detach().clone().double().to(device)
            
            layer_weights[layer_idx] = w_tensor

    # Sort keys to ensure order: Layer 0 -> Layer 1 -> ...
    sorted_indices = sorted(layer_weights.keys())
    
    weights_list = [layer_weights[i] for i in sorted_indices]
    
    print(f"Extracted {len(weights_list)} weight matrices (float64).")
    return weights_list


Ws = extract_eclipse_tensors(best_model, device='cpu')
est = LipConstEstimator(weights=Ws)
lip = est.estimate(method="ECLipsE")   
print("Lipschitz estimate for the neural controller trained with no preview and without regularization:", lip)

Ws = extract_eclipse_tensors(best_model_regNN, device='cpu')
est = LipConstEstimator(weights=Ws)
lip = est.estimate(method="ECLipsE")   
print("Lipschitz estimate for the neural controller trained with no preview and with regularization:", lip)

Ws = extract_eclipse_tensors(best_model_preview, device='cpu')
est = LipConstEstimator(weights=Ws)
lip = est.estimate(method="ECLipsE") 
print("Lipschitz estimate for the neural controller trained with preview and without regularization:", lip)

Ws = extract_eclipse_tensors(best_model_pre_regNN, device='cpu')
est = LipConstEstimator(weights=Ws)
lip = est.estimate(method="ECLipsE")   # or "ECLipsE-Fast"
print("Lipschitz estimate for the neural controller trained with preview and with regularization:", lip)




