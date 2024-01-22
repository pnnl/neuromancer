# Differentiable Control in Neuromancer

This directory contains interactive examples that can serve as a step-by-step tutorial 
showcasing control capabilities in Neuromancer.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_1_stabilize_linear_system.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Learning to stabilize a linear dynamical system.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_2_stabilize_ODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Learning to stabilize nonlinear ordinary differential equation.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_3_ref_tracking_ODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Learning constrained neural control policy for reference tracking of ordinary differential equation.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_4_NODE_control.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: Learning system Neural ODE and control policy for a dynamical systems.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_5_neural_Lyapunov.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 5: Learning neural Lyapunov function from trajectories of a nonlinear dynamical system.


## Differentiable Predictive Control in Neuromancer

[Differentiable predictive control (DPC) method](https://arxiv.org/abs/2004.11184) represents a 
flagship capability of the Neuromancer library.

DPC allows us to learn control policy parameters directly by
backpropagating model predictive control (MPC) objective function and constraints through the differentiable
 model of a dynamical system. Instances of a differentiable model include ordinary
differential equations (ODEs), including  [neural ODEs](https://arxiv.org/abs/1806.07366), 
[universal differential equations (UDEs)](https://arxiv.org/abs/2001.04385), 
or [neural state space models (SSMs)](https://ieeexplore.ieee.org/abstract/document/9482930).

The conceptual methodology shown in the figures below consists of two main steps.
In the first step, we perform system identification by learning the unknown parameters   of differentiable digital twins.
In the second step, we close the loop by combining the digital twin models with control policy, 
parametrized by neural networks, obtaining a differentiable closed-loop dynamics model.
This closed-loop model now allow us to use automatic differentiation (AD) 
to solve the parametric optimal control problem by computing the sensitivities 
of objective functions and constraints to changing problem parameters such as initial conditions, 
boundary conditions, and parametric control tasks such as time-varying reference tracking.

![DPC_abstract.](/examples/control/figs/DPC_abstract.png)  
*Conceptual methodology. Simulation of the differentiable closed-loop system dynamics 
in the forward pass is followed by backward pass computing direct policy gradients for policy optimization *

### DPC Problem Formulation

Our recent development work in Neuromancer has given us the capability to 
learn parametric control policy (parametrized by trainable weights W)

$$\mathbf{u}=\pi_{\theta}(\mathbf{x}(t), \mathbf{\xi}(t)) $$

for a given dynamical systems of the continuous time form:

$$ \frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) $$

where x(t) is the time-varying state of the considered system, u(t) are system control inputs, and f is the state
transition dynamics.  

Or in the discrete time form (e.g., obtained via ODE solver, or via state space model form):

$$ \mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$$

Formally we can formulate the DPC problem as a following parametric
optimal control problem:
![DPC_problem_form.](/examples/control/figs/DPC_problem_form.PNG)


### DPC Problem Solution

The main advantage of having a differentiable closed-loop dynamics model, control
objective function, and constraints in the DPC problem formulation
is that it allows us to use automatic
differentiation (backpropagation through time) to directly compute the policy gradient. In particular,
by representing the problem (15) as a computational graph and leveraging the chain rule, we can directly
compute the gradients of the loss function w.r.t. the policy parameters W as follows:
![DPC_policy_gradients.](/examples/control/figs/DPC_policy_gradients.PNG)  


### DPC Problem Architecture 

The forward pass of the DPC computational graph is conceptually
equivalent with a single shooting formulation of the model predictive control (MPC) problem. 
The resulting structural equivalence of the
constraints of classical implicit MPC in a dense form with DPC is illustrated in the following figure. 
Similarly to MPC, in the
open-loop rollouts, the explicit DPC policy generates future control action trajectories over N-step prediction horizon
given the feedback from the system dynamics model. Then for the closed-loop deployment, we adopt the receding
horizon control (RHC) strategy by applying only the first time step of the computed control action
![deep_MPC_var2.](/examples/control/figs/deep_MPC_var2.png)  
*Structural equivalence of DPC architecture with MPC constraints.*


### DPC Policy Optimization Algorithm

The DPC policy optimization algorithm is summarized in the following figure. 
The differentiable system dynamics model is required to instantiate the computational graph of the
DPC problem The policy gradients ∇L are obtained by differentiating the DPC loss function L over
the distribution of initial state conditions and problem parameters sampled from the given training datasets
X and Ξ, respectively. The computed policy gradients now allow us to perform direct policy optimization via
a gradient-based optimizer O. Thus the presented procedure introduces a generic approach for data-driven
solution of model-based parametric optimal control problem (15) with constrained neural control policies
![DPC_algo.](/examples/control/figs/DPC_algo.PNG)  

From a reinforcement learning (RL) perspective, the DPC loss L can be seen as a reward function,
with ∇L representing a deterministic policy gradient. The main difference compared with actor-critic
RL algorithms is that in DPC the reward function is fully parametrized by a closed-loop system dynamics
model, control objective, and constraints penalties. The model-based approach avoids approximation errors
in reward functions making DPC more sample efficient than model-free RL algorithms

### DPC Galery
The following figures illustrate DPC policy optimization algorithm on a [Part 1 example](https://github.com/pnnl/neuromancer/blob/master/examples/control/Part_1_stabilize_linear_system.py).


![cl_trajectories.](/examples/control/figs/cl_animation.gif)  
*Closed-loop trajectories of learned stabilizing neural control policy using DPC policy optimization.*

![cl_trajectories_2.](/examples/control/figs/closed%20loop%20policy%20training.gif)  
*Evolution of the closed-loop trajectories and DPC neural policy during training.*

![dpc_policy.](/examples/control/figs/policies_surfaces.png)  
*Landscapes of the learned neural policy via DPC policy optimization algorithm (right) 
and explicit MPC policy computed using parametric programming solver (left).*


### References

[1] [Drgona, J., Tuor, A., & Vrabie, D., Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees, arXiv preprint arXiv:2004.11184, 2020](https://arxiv.org/abs/2004.11184)

[2] [Drgona, Jan, et al. "Differentiable Predictive Control: An MPC Alternative for Unknown Nonlinear Systems using Constrained Deep Learning." Journal of Process Control Volume 116, August 2022, Pages 80-92](https://www.sciencedirect.com/science/article/pii/S0959152422000981) 

[3] [Drgoňa, J., Tuor, A., Skomski, E., Vasisht, S., & Vrabie, D. Deep Learning Explicit Differentiable Predictive Control Laws for Buildings. IFAC-PapersOnLine, 54(6), 14-19., 2021](https://www.sciencedirect.com/science/article/pii/S2405896321012933)

[4] [Ján Drgoňa, Sayak Mukherjee, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Learning Stochastic Parametric Differentiable Predictive Control Policies, IFAC-PapersOnLine, Volume 55, Issue 25, 2022, Pages 121-126,  ISSN 2405-8963](https://www.sciencedirect.com/science/article/pii/S2405896322015877)

[5] [Sayak Mukherjee, Ján Drgoňa, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Neural Lyapunov Differentiable Predictive Control, IEEE Conference on Decision and Control Conference 2022](https://ieeexplore.ieee.org/document/9992386/)

[6] [Wenceslao Shaw Cortez, Jan Drgona, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Differentiable Predictive Control with Safety Guarantees: A Control Barrier Function Approach, IEEE Conference on Decision and Control Conference 2022](https://ieeexplore.ieee.org/document/9993146)

[7] [Ethan King, Jan Drgona, Aaron Tuor, Shrirang Abhyankar, Craig Bakker, Arnab Bhattacharya, Draguna Vrabie, Koopman-based Differentiable Predictive Control for the Dynamics-Aware Economic Dispatch Problem, American Control Conference (ACC) 2022](https://ieeexplore.ieee.org/document/9867379)


### Cite as

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
