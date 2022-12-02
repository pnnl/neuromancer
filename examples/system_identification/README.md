## System Identification with Neural Networks in Neuromancer

Differentiable models such as Neural ordinary
differential equations (NODEs) or neural state space models (NSSMs) represent a class of black box models
that can incorporate prior physical knowledge into their architectures and loss functions. Examples include
structural assumption on the computational graph inspired by domain application, or structure of the weight
matrices of NSSM models, or networked NODE architecture. Differentiaity of NODEs and NSSMs allows us to leverage gradient-based optimization algorithms for learning the
unknown parameters of these structured digital twin models from observational data of the real system.

### System Identification Problem
Consider the non-autonomous partially observable nonlinear dynamical
system:
![sys_ID_problem.](/figs/sys_ID_problem.png)  

We assume access to a limited set of system measurements in the form of tuples, each of which corresponds
to the input-output pairs along sampled trajectories with temporal gap ∆. That is, we form a dataset
![sys_ID_dataset.](/figs/sys_ID_dataset.png)
where i = 1, 2, . . . , n represents up to n different batches of input-output trajectories with N -step time
horizon length. The primary objective of the physics-constrained system identification is to construct structured digital
twin models and learn their unknown parameters from the provided observation data to provide accurate
and robust long-term prediction capabilities.


### System Identification Models

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


### System Identification Objective with Soft Constraints 
The primary learning objective
is to minimize the mean squared error, Ly , between predicted values and the ground truth measurements
for the N -step prediction horizon:
![sys_ID_loss.](/figs/sys_ID_loss.png)

The system identification objective (11) can be augmented with various kind 
of physics-informed soft constraints. In the following we enumerate a few examples.
First, we apply inequality constraints on output predictions during training 
in order to promote the boundedness and convergence of our dynamical models:
![soft_con_sysID.](/figs/soft_con_sysID.png)  

To promote continuous trajectories of our dynamics models, we optionally apply a state smoothing loss
which minimizes the mean squared error between successive predicted states:
![state_smoothing.](/figs/state_smoothing.png)  

We include constraints penalties as additional terms to the optimization objective 14, and further define
coefficients, Q∗ as hyperparameters to scale each term in the multi-objective loss function
![sys_ID_agreggate_loss.](/figs/sys_ID_agreggate_loss.png)  


### System Identification Training Algorithm
The physics-constrained system identification training with differentiable 
digital twin models is summarized in the following Algorithm:
![sys_ID_algo.](/figs/sys_ID_algo.png)


## Neuromancer Syntax and Use
Two Neuromancer dynamics classes handle continuous time dynamics: ODEAuto and ODENonAuto. As their
names suggest, these classes handle the scenarios corresponding to Equations 1-2. Their usage is detailed
below:

### Autonomous ODEs
Autonomous ODEs are those that do not explicitly depend on time; as such, the dynamics are functions of
state variables alone. A fully-specified neural ODE of this type requires a RHS function (either a neural
network or other tensor-tensor mapping. The use of the ODEAuto class is as follows:

```python 
# Instantiate the ODE RHS as MLP:
fx = blocks.MLP(nx, nx, linear_map=slim.maps['linear'],
nonlin=activations['leakyrelu'],
hsizes=[10, 10])

# Instansiate the integrator, handing it the RHS "fx":
fxRK4 = integrators.RK4(fx, h=ts)

# Identity output mapping:
fy = slim.maps['identity'](nx, nx)

# Creating the dynamics model:
NODE = dynamics.ODEAuto(fxRK4, fy, name='dynamics',
input_key_map={"x0": f"x0_{estim.name}"})
```

Note that the transition dynamics fx is a square mapping (nx → nx) of states as expected.

### Non-Autonomous ODEs
Non-autonomous ODEs depend on time, external inputs, or both time and inputs. The syntax for these
systems changes as continuos-time representations of time and any external inputs must be available and
provided to the integrator. This is handled with the construction and passing of interpolants.

1. Time as input
An example non-autonomous system with explicit dependence on time is the Forced Duffing Oscillator, given
by the ODEs:
> <img src="https://latex.codecogs.com/svg.image?\begin{cases}\frac{dx_1}{dt}&space;=&space;x_2&space;\\\frac{dx_2}{dt}&space;=&space;-\delta&space;x_2&space;-&space;\alpha&space;x_1&space;-&space;\beta&space;x_1^3&space;&plus;&space;\gamma&space;\cos&space;(\omega&space;t)\end{cases}" title="https://latex.codecogs.com/svg.image?\begin{cases}\frac{dx_1}{dt} = x_2 \\\frac{dx_2}{dt} = -\delta x_2 - \alpha x_1 - \beta x_1^3 + \gamma \cos (\omega t)\end{cases}" />

Supposing that the model is known except for one or more of the parameters, one can build a consistent
tensor-tensor mapping to pass to an integrator and ODENonAuto class. 
First, the ODE RHS is defined:

```python
class DuffingParam(ODESystem):
    def __init__(self, insize=3, outsize=2):
    """
    :param insize:
    :param outsize:
    """
    super().__init__(insize=insize, outsize=outsize)
        self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor([0.02]), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor([8.0]), requires_grad=False)
        self.omega = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    def ode_equations(self, x):
        # states
        x0 = x[:, [0]] # (# batches,1)
        x1 = x[:, [1]]
        t = x[:, [2]]
        # equations
        dx0dt = x1
        dx1dt = -self.delta*x1 - self.alpha*x0 - self.beta*x0**3 +
            self.gamma*torch.cos(self.omega*t)
        return torch.cat([dx0dt, dx1dt], dim=-1)
```


Note that in this definition, only the paramter ω is tunable; thus, this is a 1-parameter training task.
Additionally, note the dimensionality: expected is a state dimension of three, with the third dimension
corresponding to time. The specification of the Neural ODE begins with defining a continuous representation
of time:
```python
t = torch.from_numpy(t)
interp_u = LinInterp_Offline(t, t)
```
The rest of the setup is identical to the autonomous case with the exception of the dynamics class:
```python
# Instantiate the ODE RHS:
duffing_sys = ode.DuffingParam()
# Instansiate the integrator, handing it the RHS "duffing_sys":
fxRK4 = integrators.RK4(duffing_sys, interp_u=interp_u, h=ts)
# Identity output mapping:
fy = slim.maps['identity'](nx, nx)
# Creating the dynamics model:
dynamics_model = dynamics.ODENonAuto(fxRK4, fy,
input_key_map={"x0": f"x0_{estim.name}", "Time": "Timef", 'Yf': 'Yf'},
name='dynamics', online_flag=False)
```


2. Other external inputs
Control signals are dealt with in the same manner as time: they must first be represented in a continuous
form via an interpolant. Specification of these interpolants is as follows:

```python 
t = torch.from_numpy(t) # from numpy dataset
u = raw['U'].astype(np.float32) # getting control 'u' from data dictionary
u = np.append(u,u[-1,:]).reshape(-1,2)
ut = torch.from_numpy(u)
interp_u = LinInterp_Offline(t, ut)
```

The neural ODE is specified in the same way as the forced Duffing system:

```python 
# Get the dimension of extra inputs
nu = dims['U'][1]

# Construct black-box RHS mapping from nx+nu to nx
black_box_ode = blocks.MLP(insize=nx+nu, outsize=nx, hsizes=[30, 30],
linear_map=slim.maps['linear'],
nonlin=activations['gelu'])

# Hand it over to the integrator with the interpolant:
fx_int = integrators.RK4(black_box_ode, interp_u=interp_u, h=modelSystem.ts)

# Identity output mapping:
fy = slim.maps['identity'](nx, nx)

# Creating the dynamics model:
dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
input_key_map={"x0": f"x0_{estim.name}", "Time": "Timef", 'Yf': 'Yf'},
name='dynamics', online_flag=False)
```


### System Identification Neuromancer Codebase

List of Neuromancer classes required to build DPC:

**dataset** - classes for instantiating Pytorch dataloaders with training evaluation and testing samples:
https://github.com/pnnl/neuromancer/blob/master/neuromancer/dataset.py

**dynamics** - classes parametrizing system models to be differentiated: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/dynamics.py

**estimators** - classes parametrizing estimation of the initial conditions for dynamics models: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/estimators.py

**constraints** - classes for defining constraints and custom physics-informed loss function terms: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/constraint.py

**loss** - class aggregating all instantiated constraints and loss terms 
in a scalar-valued function suitable for backpropagation-based training:
https://github.com/pnnl/neuromancer/blob/master/neuromancer/loss.py

**problem** - class agrregating trainable components (dynamics, estimators)
with loss functions in a differentiable computational graph representing 
the underlying constrained optimization problem: 
https://github.com/pnnl/neuromancer/blob/master/neuromancer/problem.py


## References

### External references
[^1]: Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential
equations. Advances in neural information processing systems, 31, 2018.
[^2]: Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational physics, 378:686–707, 2019.

### Papers using Neuromancer
+ James Koch, Zhao Chen, Aaron Tuor, Jan Drgona, Draguna Vrabie, Structural Inference of Networked Dynamical Systems with Universal Differential Equations, 	arXiv:2207.04962, (2022)
+ Drgoňa, J., Tuor, A. R., Chandan, V., & Vrabie, D. L., Physics-constrained deep learning of multi-zone building thermal dynamics. Energy and Buildings, 243, 110992, (2021)
+ E. Skomski, S. Vasisht, C. Wight, A. Tuor, J. Drgoňa and D. Vrabie, "Constrained Block Nonlinear Neural Dynamical Models," 2021 American Control Conference (ACC), 2021, pp. 3993-4000, doi: 10.23919/ACC50511.2021.9482930.
+ Skomski, E., Drgoňa, J., & Tuor, A. (2021, May). Automating Discovery of Physics-Informed Neural State Space Models via Learning and Evolution. In Learning for Dynamics and Control (pp. 980-991). PMLR.
+ Tuor, A., Drgona, J., & Vrabie, D. (2020). Constrained neural ordinary differential equations with stability guarantees. arXiv preprint arXiv:2004.10883.


## Cite as
```yaml
@article{Neuromancer2022,
  title={{NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations}},
  author={Tuor, Aaron and Drgona, Jan and Skomski, Mia and Koch, James and Chen, Zhao and Dernbach, Stefan and Legaard, Christian Møldrup and Vrabie, Draguna},
  Url= {https://github.com/pnnl/neuromancer}, 
  year={2022}
}
```