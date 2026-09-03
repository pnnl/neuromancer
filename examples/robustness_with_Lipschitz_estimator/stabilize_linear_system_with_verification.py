# -*- coding: utf-8 -*-
import torch
from torch import func
import copy
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase


from eclipse_nn.LipConstEstimator import LipConstEstimator

import random
import numpy as np

# Python built-in random
random.seed(42)

# NumPy
np.random.seed(42)

# PyTorch (CPU and GPU)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)   # if using multi-GPU



# Double integrator parameters
nx = 2
nu = 1
A = torch.tensor([[1.2, 1.0],
                  [0.0, 1.0]])
B = torch.tensor([[1.0],
                  [0.5]])

# linear state space model
xnext = lambda x, u: x @ A.T + u @ B.T    
double_integrator = Node(xnext, ['X', 'U'], ['X'], name='integrator')


# Training dataset generation
train_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# neural control policy
mlp = blocks.MLP(nx, nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
policy = Node(mlp, ['X'], ['U'], name='policy')

# closed loop system definition
cl_system = System([policy, double_integrator])
# cl_system.show()

# Define optimization problem
u = variable('U')
x = variable('X')
action_loss = 0.0001 * (u == 0.)^2  # control penalty
regulation_loss = 10. * (x == 0.)^2  # target position
loss = PenaltyLoss([action_loss, regulation_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)


trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=1000,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)


# Train model with prediction horizon of 2
cl_system.nsteps = 2
best_model = trainer.train()

## We warm-start from the standard trained policy and fine-tune with a Jacobian penalty on u=NN(x)
# reuse the trained NN
policy_reg = copy.deepcopy(policy)
model = policy_reg.callable

def jac_calculator(X):
    batch_jac = func.vmap(
        func.jacrev(lambda x_single: model(x_single.unsqueeze(0)).view(-1))
    )(X)  # [Batch, nu, nx]
    return torch.linalg.norm(batch_jac, ord='fro', dim=(-2, -1)).unsqueeze(-1)  # [Batch, 1]

jac_fro = Node(jac_calculator, ['X'], ['Jfro'], name='jac_fro')

# closed loop system definition (add the jac_fro node)
cl_system_reg = System([policy_reg, double_integrator, jac_fro])


# variables
u = variable('U')
x = variable('X')
j = variable('Jfro')   # from the Jacobian-Fro node

action_loss     =  0.0001 * (u == 0.)^2       
regulation_loss = 10. * (x == 0.)^2     
jac_loss    = 0.1 * (j == 0.)^2 

loss_reg = PenaltyLoss([action_loss, regulation_loss,jac_loss], [])
problem_reg = Problem([cl_system_reg], loss_reg)

optimizer = torch.optim.AdamW(policy_reg.parameters(), lr=0.0003)


trainer_reg = Trainer(
    problem_reg,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)


# Train model with prediction horizon of 2
cl_system_reg.nsteps = 2
best_model_reg = trainer_reg.train()


## We warm-start from the standard trained policy and fine-tune with a Jacobian penalty on the closed-loop map x -> f(x,NN(x))
# reuse the trained NN
policy_reg_cl = copy.deepcopy(policy)
model_cl = policy_reg_cl.callable

def jac_cl_calculator(X):
    batch_j_pi = func.vmap(
        func.jacrev(lambda x_single: model_cl(x_single.unsqueeze(0)).view(-1))
    )(X)  # [Batch, nu, nx]

    A_ = A.to(device=X.device, dtype=X.dtype)
    B_ = B.to(device=X.device, dtype=X.dtype)

    J_cl = A_ + torch.matmul(B_, batch_j_pi)  # [Batch, nx, nx]
    return torch.linalg.norm(J_cl, ord='fro', dim=(-2, -1)).unsqueeze(-1)  # [Batch, 1]

jac_cl_node = Node(jac_cl_calculator, ['X'], ['Jcl_fro'], name='jac_cl')

# closed loop system definition (add the jac_cl_node)
cl_system_reg_cl = System([policy_reg_cl, double_integrator, jac_cl_node])

# variables
u = variable('U')
x = variable('X')
jcl = variable('Jcl_fro')   # from the Jacobian-Fro node

action_loss     =  0.0001 * (u == 0.)^2       
regulation_loss = 10. * (x == 0.)^2     
jac_cl_loss    = 4 * (jcl == 0.)^2 

loss_reg_cl = PenaltyLoss([action_loss, regulation_loss,jac_cl_loss], [])
problem_reg_cl = Problem([cl_system_reg_cl], loss_reg_cl)


optimizer = torch.optim.AdamW(policy_reg_cl.parameters(), lr=0.0003)

trainer_reg_cl = Trainer(
    problem_reg_cl,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)


# Train model with prediction horizon of 2
cl_system_reg_cl.nsteps = 2
best_model_reg_cl = trainer_reg_cl.train()


## We evaluate all three controllers on a longer rollout horizon and compare state trajectories and control inputs
# Test best model with prediction horizon of 30
problem.load_state_dict(best_model)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system.nsteps = nsteps
trajectories = cl_system(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')

# Test best model with prediction horizon of 50
problem_reg.load_state_dict(best_model_reg)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system_reg.nsteps = nsteps
trajectories = cl_system_reg(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')

# Test best model with prediction horizon of 50
problem_reg_cl.load_state_dict(best_model_reg_cl)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system_reg_cl.nsteps = nsteps
trajectories = cl_system_reg_cl(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')

## Now we estimate the Lipschitz constant of all the three neural controllers u=NN(x).
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
lip = est.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller from standard training is {lip}")



Ws_reg = extract_eclipse_tensors(best_model_reg, device='cpu')
est_reg = LipConstEstimator(weights=Ws_reg)
lip_reg = est_reg.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller trained with regularization on the Jacobian of policy u=NN(x)  is {lip_reg}")



Ws_reg_cl = extract_eclipse_tensors(best_model_reg_cl, device='cpu')
est_reg_cl = LipConstEstimator(weights=Ws_reg_cl)
lip_reg_cl = est_reg_cl.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller trained with regularization on the Jacobian of closed-loop system  is {lip_reg_cl}")