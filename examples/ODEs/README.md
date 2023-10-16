# System Identification for ODEs in Neuromancer

This directory contains interactive examples that can serve as a step-by-step tutorial 
showcasing system identification capabilities for ordinary differential equations (ODEs) in Neuromancer.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_1_NODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Neural Ordinary Differential Equations (NODEs)

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_2_param_estim_ODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Parameter estimation of ODE system

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_3_UDE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Universal Differential Equations (UDEs)

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_4_nonauto_NODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: NODEs with exogenous inputs

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_5_nonauto_NSSM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 5: Neural State Space Models (NSSMs) with exogenous inputs

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_6_NetworkODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 6: Data-driven modeling of resistance-capacitance (RC) network ODEs

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_7_DeepKoopman.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 7: Deep Koopman operator

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_8_nonauto_DeepKoopman.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 8: control-oriented Deep Koopman operator


## System Identification

[System identification](https://en.wikipedia.org/wiki/System_identification) is using statistical methods to construct mathematical models of dynamical systems given the measured observations of the system behavior.

### System ID methods

In this library, we are primarily interested in differentiable system ID methods that can incorporate prior physical knowledge into their architectures and loss functions. Examples include
structural assumption on the computational graph inspired by domain application, structure of the weight
matrices, or network architectures. Differentiaity allows us to 
leverage gradient-based optimization algorithms for learning the
unknown parameters of these structured digital twin models from observational data of the real system.

Neuromancer currently supports the following system identification methods:
+ [Neural ordinary differential equations (NODEs)](https://arxiv.org/abs/1806.07366)
+ [Neural state space models (NSSMs)](https://arxiv.org/abs/2011.13497)
+ [Universal differential equations (UDEs)](https://arxiv.org/abs/2001.04385)
  

### System Identification Models

Our recent development work in Neuromancer has given us the capability 
to learn discrete time dynamical systems (Neural State Space Models) in the following forms:

$$x_{k+1}=f_{\theta}(x_k)$$  

$$x_{k+1}=f_{\theta}(x_k, u_k)$$

as well as continuous dynamical systems (Neural ODEs) in the following forms:

$$\frac{dx}{dt}=f_{\theta}(x(t))$$  

$$\frac{dx}{dt}=f_{\theta}(x(t), t)$$  

$$\frac{dx}{dt}=f_{\theta}(x(t), u(t), t)$$  

Where x(t) is the time-varying state of the considered system, u(t) are system control inputs, and f is the state
transition dynamics. This modeling strategy can be thought of as an equivalent method to Neural Ordinary
Differential Equations[1], whereby an ODE of the above forms is fit to data with a universal function
approximator (e.g., deep neural network) acting as the state transition dynamics. To train an appropriate
RHS, Chen et al. utilize a continuous form of the adjoint equation, itself solved with an ODESolver.
An alternative is to unroll the ODE solvers and utilize the autodifferentiation properties of PyTorch 
to build differentiable canonical ODE integrators (e.g. as in Raissi et al.[2]).

In the case of the continuous-time model, we need to first integrate the ODE system using an ODE solver, 
e.g., [Euler](https://en.wikipedia.org/wiki/Euler_method), 
or [Runge–Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods), to obtain the discretized system:

$$x_{k+1} = \text{ODESolve}(f_{\theta}(x_k))$$ 

### System Identification Dataset
Consider the following ordinary differential equation system:

$$\frac{dx}{dt} = f(x(t))$$ 
Where $x(t)$ represent system states and $\frac{dx}{dt}$ its time derivative. The dynamics 
of the model is prescribed by the ODE equations $f(\cdot)$

We assume access to a limited set of system measurements of the system states 
in the form of sampled trajectories. That is, we form a dataset:
$$\hat{X} = [\hat{x}^i_0, ..., \hat{x}^i_{N}], \, \, i \in [1, ..., m]$$
where $N$ represents the prediction horizon, $m$ represents number of measured trajectories, 
and $i$ represents an index of the sampled trajectory.

### System Identification Problem

The primary objective of the system identification task is to learn the unknown parameters $\theta$
of the model $f_{\theta}(x(t)) \approx f(x(t))$ approximating the real system.

Standard learning objective
is to minimize the mean squared error between predicted values and the ground truth measurements:

$$\ell_x = Q_x||x^i_k - \hat{x}^i_k||_2^2$$ 

The primary objective  can be augmented with various kinds 
of physics-informed soft constraints. For instance, one could impose finite difference loss
matching the model's and data's first derivative estimates.

$$\ell_{dx} =  Q_{dx}||\Delta x^i_k - \Delta \hat{x}^i_k||_2^2$$

where $\Delta x^i_k = x^i_{k+1} - x^i_k$

There are many more loss function terms that could be included in the system ID task that can be combined 
in the final system ID loss function:
 
$$ \text{min}  \sum_{i=1}^m \Big( \sum_{k=1}^{N}  \ell_x  +  \sum_{k=1}^{N-1} \ell_{dx} \Big) $$

$$ \text{s.t.}  &emsp;  x^i_{k+1} =  \text{ODESolve}(f_{\theta}(x^i_k)) $$  



## Neuromancer Syntax and Use

Neuromancer provides an intuitive API for defining dynamical system models. 

### Neural State Space Models in Neuromancer
```python 
# Instantiate the state space model x_k+1 = Ax_k + Bu_k with given matrices A, B
fx_ssm = lambda x, u: x @ A.T + u @ B.T

# Creating the dynamics model to be trained over a given prediction horizon:
dynamics_model = System([Node(fx_ssm, ['xn'], ['xn'])], nsteps=10)
```

### Neural ODEs in Neuromancer
```python 
# Instantiate the ODE RHS as MLP:
fx = blocks.MLP(nx, nx, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[60, 60])

# Instansiate the integrator with given step size h, handing it the RHS "fx":
fxRK4 = integrators.RK4(fx, h=1.0)

# Creating the dynamics model to be trained over a given prediction horizon:
dynamics_model = System([Node(fxRK4, ['xn'], ['xn'])], nsteps=10)
```

### System Identification Neuromancer Codebase

List of Neuromancer classes required to build DPC:

**dataset** - classes for instantiating Pytorch dataloaders with training evaluation and testing samples:
https://github.com/pnnl/neuromancer/blob/master/src/neuromancer/dataset.py

**dynamics** - set of modules for constructing a wide-class of dynamical system models: 
https://github.com/pnnl/neuromancer/tree/master/src/neuromancer/dynamics

**constraints** - classes for defining constraints and custom physics-informed loss function terms: 
https://github.com/pnnl/neuromancer/blob/master/src/neuromancer/constraint.py

**loss** - class aggregating all instantiated constraints and loss terms 
in a scalar-valued function suitable for backpropagation-based training:
https://github.com/pnnl/neuromancer/blob/master/src/neuromancer/loss.py

**problem** - class agrregating trainable components (dynamics, estimators)
with loss functions in a differentiable computational graph representing 
the underlying constrained optimization problem: 
https://github.com/pnnl/neuromancer/blob/master/src/neuromancer/problem.py


## References

### Alternative packages
It is important to note that there are two dominant neural ODE packages freely
available. The first is DiffEqFlux.jl developed and maintained by SciML within the Julia ecosystem. The
second is torchdyn which lives within the PyTorch ecosystem. Both packages are well-documented and have
become established in application-based research literature.

### External references
+ [1]: [Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential
equations. Advances in neural information processing systems, 31, 2018.](https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)
+ [2]: [Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational physics, 378:686–707, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)

### Papers using Neuromancer
+ [J. Koch, Z. Chen, A. Tuor, J. Drgona, D. Vrabie; Structural inference of networked dynamical systems with universal differential equations. Chaos 1 February 2023; 33 (2): 023103. https://doi.org/10.1063/5.0109093](https://pubs.aip.org/aip/cha/article-abstract/33/2/023103/2875955/Structural-inference-of-networked-dynamical?redirectedFrom=fulltext)
+ [Drgoňa, J., Tuor, A. R., Chandan, V., & Vrabie, D. L., Physics-constrained deep learning of multi-zone building thermal dynamics. Energy and Buildings, 243, 110992, (2021)](https://www.sciencedirect.com/science/article/pii/S0378778821002760)
+ [E. Skomski, S. Vasisht, C. Wight, A. Tuor, J. Drgoňa and D. Vrabie, "Constrained Block Nonlinear Neural Dynamical Models," 2021 American Control Conference (ACC), 2021, pp. 3993-4000, doi: 10.23919/ACC50511.2021.9482930.](https://ieeexplore.ieee.org/document/9482930)
+ [Skomski, E., Drgoňa, J., & Tuor, A. (2021, May). Automating Discovery of Physics-Informed Neural State Space Models via Learning and Evolution. In Learning for Dynamics and Control (pp. 980-991). PMLR.](https://proceedings.mlr.press/v144/skomski21a.html)
+ [Tuor, A., Drgona, J., & Vrabie, D. (2020). Constrained neural ordinary differential equations with stability guarantees. arXiv preprint arXiv:2004.10883.](https://arxiv.org/abs/2004.10883)


