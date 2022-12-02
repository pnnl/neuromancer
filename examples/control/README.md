# Differentiable Predictive Control in Neuromancer
Differentiable predictive control (DPC) method allows us to learn control policy parameters directly by
backpropagating model predictive control (MPC) objective function and constraints through the differentiable
digital twin model of a dynamical system.

The conceptual methodology shown in the figures below consists of two main steps.
In the first step, we perform system identification by learning the unknown parameters of differentiable digital twins.
In the second step, we close the loop by combining the digital twin models with control policy, 
parametrized by neural networks, obtaining a differentiable closed-loop dynamics model.
This closed-loop model now allow us to use automatic differentiation (AD) 
to solve the parametric optimal control problem by computing the sensitivities 
of objective functions and constraints to changing problem parameters such as initial conditions, 
boundary conditions, and parametric control tasks such as time-varying reference tracking.

![DPC_abstract.](/figs/DPC_abstract.png)  
*Conceptual methodology. Simulation of the differentiable closed-loop system dynamics 
in the forward pass is followed by backward pass computing direct policy gradients for policy optimization *

## DPC Problem Formulation

Our recent development work in Neuromancer has given us the capability to 
learn parametric control policy (parametrized by trainable weights W)

> <img src="https://latex.codecogs.com/svg.image?\mathbf{u}=\pi_{W}(\mathbf{x}(t), \mathbf{\xi}(t))" title="https://latex.codecogs.com/svg.image?\mathbf{u}=\pi_{\theta}(\mathbf{x}(t), \mathbf{\xi}(t))" />

for a given dynamical systems of the continuous time form:

> <img src="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t),&space;\mathbf{u}(t))" title="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t), \mathbf{u}(t))" />

where x(t) is the time-varying state of the considered system, u(t) are system control inputs, and f is the state
transition dynamics.  

Or in the discrete time form (e.g., obtained via ODE solver, or via state space model form):

> <img src="https://latex.codecogs.com/svg.image?\mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_k,&space;\mathbf{u}_k)" title="https://latex.codecogs.com/svg.image?\mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_k,&space;\mathbf{u}_k)" />

Formally we can formulate the DPC problem as a following parametric
optimal control problem:
![DPC_problem_form.](/figs/DPC_problem_form.png)  
*DPC problem formulation.*


## DPC Problem Solution

The main advantage of having a differentiable closed-loop dynamics model, control
objective function, and constraints in the DPC problem formulation
is that it allows us to use automatic
differentiation (backpropagation through time) to directly compute the policy gradient. In particular,
by representing the problem (15) as a computational graph and leveraging the chain rule, we can directly
compute the gradients of the loss function w.r.t. the policy parameters W as follows:
![DPC_policy_gradients.](/figs/DPC_policy_gradients.png)  
*DPC policy gradients.*

## DPC Problem Architecture 

The forward pass of the DPC computational graph is conceptually
equivalent with a single shooting formulation of the model predictive control (MPC) problem. 
The resulting structural equivalence of the
constraints of classical implicit MPC in a dense form with DPC is illustrated in the following figure. 
Similarly to MPC, in the
open-loop rollouts, the explicit DPC policy generates future control action trajectories over N-step prediction horizon
given the feedback from the system dynamics model. Then for the closed-loop deployment, we adopt the receding
horizon control (RHC) strategy by applying only the first time step of the computed control action
![deep_MPC_var2.](/figs/deep_MPC_var2.png)  
*Structural equivalence of DPC architecture with MPC constraints.*


## DPC Policy Optimization Algorithm

The DPC policy optimization algorithm is summarized in the following figure. 
The differentiable system dynamics model is required to instantiate the computational graph of the
DPC problem The policy gradients ∇L are obtained by differentiating the DPC loss function L over
the distribution of initial state conditions and problem parameters sampled from the given training datasets
X and Ξ, respectively. The computed policy gradients now allow us to perform direct policy optimization via
a gradient-based optimizer O. Thus the presented procedure introduces a generic approach for data-driven
solution of model-based parametric optimal control problem (15) with constrained neural control policies
![DPC_algo.](/figs/DPC_algo.png)  
*DPC policy optimization algorithm.*

From a reinforcement learning (RL) perspective, the DPC loss L can be seen as a reward function,
with ∇L representing a deterministic policy gradient. The main difference compared with actor-critic
RL algorithms is that in DPC the reward function is fully parametrized by a closed-loop system dynamics
model, control objective, and constraints penalties. The model-based approach avoids approximation errors
in reward functions making DPC more sample efficient than model-free RL algorithms

## DPC Syntax and Usage
The following code illustrates the implementation of Differentiable
Predictive Control in Neuromancer:
```python 
# Tutorial example for Differentiable Predictive Control 

import torch
import neuromancer as nm
import slim 
import numpy as np

"""
# # #  Dataset 
"""
# problem dimensions
nx = 2
ny = 2
nu = 1
# constraints bounds
umin = -1.
umax = 1.
xmin = -2.
xmax = 2.
xN_min = -0.1
xN_max = 0.1
# number of prediction timesteps
nsteps = 1
# number of datapoints to sample
nsim = 10000
#  Randomly sampled input output trajectories for training. We treat states as observables, i.e. Y = X
sequences = {"Y": 3*np.random.randn(nsim, nx)}
nstep_data, loop_data, dims = nm.dataset.get_sequence_dataloaders(sequences, nsteps)
train_data, dev_data, test_data = nstep_data
train_loop, dev_loop, test_loop = loop_data

"""
# # #  System model and Control policy
"""
# Fully observable estimator as identity map: x0 = Yp[-1]
# x_0 = e(Yp)
# Yp = [y_-N, ..., y_0]
estimator = nm.estimators.FullyObservable(
               {**dims, "x0": (nx,)},
               input_keys=["Yp"], name='est')
               
# full state feedback control policy
# [u_0, ..., u_N] = p(x_0)
policy = nm.policies.MLPPolicy(
    {estimator.output_keys[0]: (nx,), 'U': (nsim, nu)},
    nsteps=1,
    linear_map=slim.maps['linear'],
    nonlin=nm.activations.activations['relu'],
    hsizes=[20] * 4,
    input_keys=[estimator.output_keys[0]],
    name='pol',
)

# LTI SSM
# x_k+1 = Ax_k + Bu_k
# y_k+1 = Cx_k+1
A = torch.tensor([[1.2, 1.0],
                  [0.0, 1.0]])
B = torch.tensor([[1.0],
                  [0.5]])
C = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])
dynamics_model = nm.dynamics.LinearSSM(A, B, C, name='mod',
                          input_key_map={'x0': estimator.output_keys[0],
                                         'Uf': policy.output_keys[0]})
dynamics_model.requires_grad_(False)  # fix model parameters

"""
# # #  DPC objectives and constraints
"""
u = nm.constraint.variable(policy.output_keys[0])
y = nm.constraint.variable(dynamics_model.output_keys[2])
# objective weights
Qu = 0.1
Qx = 1.
Q_con_x = 10.
Q_con_u = 100.
Qn = 1.
# objectives
action_loss = Qu * ((u == 0.) ^ 2)  # control penalty
regulation_loss = Qx * ((y == 0.) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = Q_con_x * (y > xmin)
state_upper_bound_penalty = Q_con_x * (y < xmax)
inputs_lower_bound_penalty = Q_con_u * (u > umin)
inputs_upper_bound_penalty = Q_con_u * (u < umax)
terminal_lower_bound_penalty = Qn * (y[:, [-1], :] > xN_min)
terminal_upper_bound_penalty = Qn * (y[:, [-1], :] < xN_max)


# list of objectives and constraints
objectives = [regulation_loss, action_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
    inputs_lower_bound_penalty,
    inputs_upper_bound_penalty,
    terminal_lower_bound_penalty,
    terminal_upper_bound_penalty,
]

"""
# # #  DPC problem = objectives + constraints + trainable components 
"""
# data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
components = [estimator, policy, dynamics_model]
# create constrained optimization loss
loss = nm.loss.PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = nm.problem.Problem(components, loss)
# plot computational graph
problem.plot_graph()

"""
# # #  DPC trainer 
"""
optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
trainer = nm.trainer.Trainer(
    problem,
    train_data,
    dev_data,
    test_data,
    optimizer,
    epochs=1000,
    train_metric="nstep_train_loss",
    dev_metric="nstep_dev_loss",
    test_metric="nstep_test_loss",
    eval_metric='nstep_dev_loss',
)
# Train control policy
best_model = trainer.train()
best_outputs = trainer.test(best_model)
```


List of Neuromancer classes required to build DPC:

**dataset** - classes for instantiating Pytorch dataloaders with training evaluation and testing samples:
https://github.com/pnnl/neuromancer/blob/master/neuromancer/dataset.py

**policies** - classes parametrizing control feedback laws: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/policies.py

**dynamics** - classes parametrizing system models to be differentiated: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/dynamics.py

**estimators** - classes parametrizing estimation of the initial conditions for dynamics models: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/estimators.py

**constraints** - classes for defining constraints and custom physics-informed loss function terms: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/constraint.py

**loss** - class aggregating all instantiated constraints and loss terms 
in a scalar-valued function suitable for backpropagation-based training:
https://github.com/pnnl/neuromancer/blob/master/neuromancer/loss.py

**problem** - class agrregating trainable components (policies, dynamics, estimators)
with loss functions in a differentiable computational graph representing 
the underlying constrained optimization problem: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/problem.py


## DPC Examples

This folder demonstrates a few examples for training explicit neural control policies
given dynamical system model in the state space model (SSM) and in the neural ODE form.

![cl_trajectories.](/figs/cl_animation.gif)  
*Example 1: Closed-loop trajectories of learned stabilizing neural control policy using DPC policy optimization.*

![cl_trajectories_2.](/figs/closed%20loop%20policy%20training.gif)  
*Example 1: Evolution of the closed-loop trajectories and DPC neural policy during training.*

![dpc_policy.](/figs/policies_surfaces.png)  
*Example 1: Landscapes of the learned neural policy via DPC policy optimization algorithm (right) 
and explicit MPC policy computed using parametric programming solver (left).*


## References

[1] Drgona, J., Tuor, A., & Vrabie, D., Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees, arXiv preprint arXiv:2004.11184, 2020

[2] Drgona, Jan, et al. "Differentiable Predictive Control: An MPC Alternative for Unknown Nonlinear Systems using Constrained Deep Learning." Journal of Process Control Volume 116, August 2022, Pages 80-92 

[3] Drgoňa, J., Tuor, A., Skomski, E., Vasisht, S., & Vrabie, D. Deep Learning Explicit Differentiable Predictive Control Laws for Buildings. IFAC-PapersOnLine, 54(6), 14-19., 2021

[4] Ján Drgoňa, Sayak Mukherjee, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Learning Stochastic Parametric Differentiable Predictive Control Policies, arXiv:2205.10728, accepted for IFAC ROCOND conference 2022

[5] Sayak Mukherjee, Ján Drgoňa, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Neural Lyapunov Differentiable Predictive Control, IEEE Conference on Decision and Control Conference 2022

[6] Wenceslao Shaw Cortez, Jan Drgona, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Differentiable Predictive Control with Safety Guarantees: A Control Barrier Function Approach, IEEE Conference on Decision and Control Conference 2022

[7] Ethan King, Jan Drgona, Aaron Tuor, Shrirang Abhyankar, Craig Bakker, Arnab Bhattacharya, Draguna Vrabie, Koopman-based Differentiable Predictive Control for the Dynamics-Aware Economic Dispatch Problem, American Control Conference (ACC) 2022


## Cite as

```yaml
@misc{drgona2022_DPC,
      title={Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees}, 
      author={Jan Drgona and Aaron Tuor and Draguna Vrabie},
      year={2022},
      eprint={2004.11184},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
}
```


```yaml
@article{DRGONA202280,
        title = {{Differentiable predictive control {:} Deep learning alternative to explicit model predictive control for unknown nonlinear systems}},
        journal = {Journal of Process Control},
        volume = {116},
        pages = {80-92},
        year = {2022},
        issn = {0959-1524},
        author = {Ján Drgoňa and Karol Kiš and Aaron Tuor and Draguna Vrabie and Martin Klaučo}
}
```

```yaml
@misc{drgona2022_SDPC,
      title={Learning Stochastic Parametric Differentiable Predictive Control Policies}, 
      author={Jan Drgona and Sayak Mukherjee and Aaron Tuor and Mahantesh Halappanavar and Draguna Vrabie},
      year={2022},
      eprint={2203.01447},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
}
```
