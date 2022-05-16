# Continuous Time Integration in Neuromancer
Our recent development work in Neuromancer has given us the capability to learn dynamical systems of the form:

> <img src="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t))" title="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t))" />

or

> <img src="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t),&space;t)" title="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t), t)" />

or

> <img src="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t),&space;\mathbf{u}(t),&space;t)" title="https://latex.codecogs.com/svg.image?\frac{d\mathbf{x}(t)}{dt}=\mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t)" />

where x(t) is the time-varying state of the considered system, u(t) are system control inputs, and f is the state
transition dynamics. This modeling strategy can be thought of as an equivalent method to Neural Ordinary
Differential Equations[^1], whereby an ODE of the above forms is fit to data with a universal function
approximator (e.g. deep neural network) acting as the state transition dynamics. To train an appropriate
RHS, Chen et al. utilize a continuous form of the adjoint equation; itself solved with an ODESolver.
Instead, we choose to utilize the autodifferentiation properties of PyTorch to build differentiable canonical
ODE integrators (e.g. as in Raissi et al.[^2]).

We wish to test the capability of this methodology in a variety of situations and configurations. Of
particular interest is the predictive capability of this class of methods compared with Neural State Space
Models and other traditional “black-box” modeling techniques.

Before moving on, it is important to note that there are two dominant neural ODE packages freely
available. The first is DiffEqFlux.jl developed and maintained by SciML within the Julia ecosystem. The
second is torchdyn which lives within the PyTorch ecosystem. Both packages are well-documented and have
become established in application-based research literature.

## Syntax and Usage
Two Neuromancer dynamics classes handle continuous time dynamics: ODEAuto and ODENonAuto. As their
names suggest, these classes handle the scenarios corresponding to Equations 1-2. Their usage is detailed
below:

### Autonomous ODEs
Autonomous ODEs are those that do not explicitly depend on time; as such, the dynamics are functions of
state variables alone. A fully-specified neural ODE of this type requires a RHS function (either a neural
network or other tensor-tensor mapping. The use of the ODEAuto class is as follows:

![Image](auto_ODE.jpy)

[^1]: Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential
equations. Advances in neural information processing systems, 31, 2018.
[^2]: Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational physics, 378:686–707, 2019.
