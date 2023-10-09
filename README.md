# NeuroMANCER v1.4.1

**Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations (NeuroMANCER)**
is an open-source differentiable programming (DP) library for solving parametric constrained optimization problems, 
physics-informed system identification, and parametric model-based optimal control.
NeuroMANCER is written in [PyTorch](https://pytorch.org/) and allows for systematic 
integration of machine learning with scientific computing for creating end-to-end 
differentiable models and algorithms embedded with prior knowledge and physics.


### New in v1.4.1
We've made some backwards-compatible changes in order to simplify integration and support multiple symbolic inputs to `nn.Modules` in our `blocks` interface.

**New Colab Examples:**  
> ⭐ [Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs) in NeuroMANCER](#physics-informed-neural-networks-pinns-for-partial-differential-equations-pdes)

> ⭐ [System identification for ordinary differential equations (ODEs)](#ordinary-differential-equations-odes)

See [v1.4.1 release notes](#version-141-release-notes) for more details.

## Features and Examples

Extensive set of tutorials can be found in the 
[examples](https://github.com/pnnl/neuromancer/tree/master/examples) folder.
Interactive notebook versions of examples are available on Google Colab!
Test out NeuroMANCER functionality before cloning the repository and setting up an
environment.

### Intro to NeuroMANCER

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/tutorials/part_1_linear_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 1: Linear regression in PyTorch vs NeuroMANCER.  

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/tutorials/part_2_variable.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 2: NeuroMANCER syntax tutorial: variables, constraints, and objectives.  

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/tutorials/part_3_node.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 3: NeuroMANCER syntax tutorial: modules, Node, and System class.


### Parametric Programming

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_1_basics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 1: Learning to solve a constrained optimization problem.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_2_pQP.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 2: Learning to solve a quadratically-constrained optimization problem.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_3_pNLP.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 3: Learning to solve a set of 2D constrained optimization problems.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_4_projectedGradient.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 
Part 4: Learning to solve a constrained optimization problem with projected gradient method.

### Ordinary Differential Equations (ODEs)
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


### Physics-Informed Neural Networks (PINNs) for Partial Differential Equations (PDEs)
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_1_PINN_DiffusionEquation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Diffusion Equation
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_2_PINN_BurgersEquation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Burgers' Equation
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_3_PINN_BurgersEquation_inverse.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Burgers' Equation w/ Parameter Estimation (Inverse Problem)

### Control

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_1_stabilize_linear_system.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Learning to stabilize a linear dynamical system.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_2_stabilize_ODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Learning to stabilize a nonlinear differential equation.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_3_ref_tracking_ODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Learning to control a nonlinear differential equation.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_4_NODE_control.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: Learning neural ODE model and control policy for an unknown dynamical system.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/Part_5_neural_Lyapunov.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 5: Learning neural Lyapunov function for a nonlinear dynamical system.



## Documentation
The documentation for the library can be found [online](https://pnnl.github.io/neuromancer/). 
There is also an [introduction video](https://www.youtube.com/watch?v=YkFKz-DgC98) covering 
core features of the library. 


```python 
# Neuromancer syntax example for constrained optimization
import neuromancer as nm
import torch 

# define neural architecture 
func = nm.modules.blocks.MLP(insize=1, outsize=2, 
                             linear_map=nm.slim.maps['linear'], 
                             nonlin=torch.nn.ReLU, hsizes=[80] * 4)
# wrap neural net into symbolic representation via the Node class: map(p) -> x
map = nm.system.Node(func, ['p'], ['x'], name='map')
    
# define decision variables
x = nm.constraint.variable("x")[:, [0]]
y = nm.constraint.variable("x")[:, [1]]
# problem parameters sampled in the dataset
p = nm.constraint.variable('p')

# define objective function
f = (1-x)**2 + (y-x**2)**2
obj = f.minimize(weight=1.0)

# define constraints
con_1 = 100.*(x >= y)
con_2 = 100.*(x**2+y**2 <= p**2)

# create penalty method-based loss function
loss = nm.loss.PenaltyLoss(objectives=[obj], constraints=[con_1, con_2])
# construct differentiable constrained optimization problem
problem = nm.problem.Problem(nodes=[map], loss=loss)
```

![UML diagram](figs/class_diagram.png)
*UML diagram of NeuroMANCER classes.*


## Installation

For either pip or conda installation, first clone the neuromancer package.
A dedicated virtual environment (conda or otherwise) is recommended. 

Note: If you have a previous neuromancer env it would be best at this point to create a new environment given the following instructions.

```bash
git clone -b master https://github.com/pnnl/neuromancer.git --single-branch
```

### PIP Install
Recommended installation.  
Pip installation is broken up into required dependencies for core Neuromancer
and dependencies associated with the examples, tests, and generating the documentation.
Below we give instructions to install all dependencies in a conda virtual enviroment via pip. 
You need at least pip version >= 21.3.

#### Create and activate virtual environment

``` bash
conda create -n neuromancer python=3.10.4
conda activate neuromancer
```

#### Install neuromancer and all dependencies.
From top level directory of cloned neuromancer run:

```bash
pip install -e.[docs,tests,examples]
```

OR, for zsh users:
```zsh
pip install -e.'[docs,tests,examples]'
```

See the `pyproject.toml` file for reference.

``` toml
[project.optional-dependencies]
tests = ["pytest", "hypothesis"]
examples = ["casadi", "cvxpy", "imageio"]
docs = ["sphinx", "sphinx-rtd-theme"]
```

#### Note on pip install with `examples` on MacOS (Apple M1)
Before CVXPY can be installed on Apple M1, you must install `cmake` via Homebrew:

```zsh
brew install cmake
```

See [CVXPY installation instructions](https://www.cvxpy.org/install/index.html) for more details.


### Conda install
Conda install is recommended for GPU acceleration. 
In many cases the following simple install should work for the specified OS

#### Create environment & install dependencies
##### Ubuntu

``` bash
conda env create -f linux_env.yml
conda activate neuromancer
```

##### Windows

``` bash
conda env create -f windows_env.yml
conda activate neuromancer
conda install -c defaults intel-openmp -f
```

##### MacOS (Apple M1)

``` bash
conda env create -f osxarm64_env.yml
conda activate neuromancer
```

##### Other (manually install all dependencies)

!!! Pay attention to comments for non-Linux OS !!!

``` bash
conda create -n neuromancer python=3.10.4
conda activate neuromancer
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
## OR (for Mac): conda install pytorch -c pytorch
conda config --append channels conda-forge
conda install scipy numpy"<1.24.0" matplotlib scikit-learn pandas dill mlflow pydot=1.4.2 pyts numba
conda install networkx=3.0 plum-dispatch=1.7.3 
conda install -c anaconda pytest hypothesis
conda install cvxpy cvxopt casadi seaborn imageio
conda install tqdm torchdiffeq toml
## (for Windows): conda install -c defaults intel-openmp -f
```

#### Install NeuroMANCER package
From the top level directory of cloned neuromancer
(in the activated environment where the dependencies have been installed):

```bash
pip install -e . --no-deps
```

### Test NeuroMANCER install
Run pytest on the [tests folder](https://github.com/pnnl/neuromancer/tree/master/tests). 
It should take about 2 minutes to run the tests on CPU. 
There will be a lot of warnings that you can safely ignore. These warnings will be cleaned 
up in a future release.


## Community Development

We welcome contributions and feedback from the open-source community!

### Discussions

[Discussions](https://github.com/pnnl/neuromancer/discussions) should be the first line of contact for new users to provide direct feedback on the library.
Post your [Ideas](https://github.com/pnnl/neuromancer/discussions/categories/ideas) for new features or examples, 
showcase your work using neuromancer in [Show and tell](https://github.com/pnnl/neuromancer/discussions/categories/show-and-tell),
or get support for usage in [Q&As](https://github.com/pnnl/neuromancer/discussions/categories/q-a),
please post them in one of our  categories.


### Contributing examples
If you have an example of using NeuroMANCER to solve an interesting problem, or of using 
NeuroMANCER in a unique way please share them in [Show and tell](https://github.com/pnnl/neuromancer/discussions/categories/show-and-tell)
discussions.
The best examples might be incorporated into our current library of examples. 
To submit an example, create a folder for your example/s in the example folder if there isn't 
currently an applicable folder and place either your executable python file or notebook file there. 
Push your code back to Github and then submit a pull request. Please make sure to note in a comment at 
the top of your code if there are additional dependencies to run your example and how to install 
those dependencies. 

### Contributing code

We welcome contributions to NeuroMANCER. Please accompany contributions with some lightweight unit tests
via pytest (see test/ folder for some examples of easy to compose unit tests using pytest). 
In addition to unit tests
a script utilizing introduced new classes or modules should be placed in the examples folder. 
To contribute a new well-developed feature please submit a pull request (PR). 
Before creating a PR, we encourage developers to discuss and document the intended feature
in [Ideas](https://github.com/pnnl/neuromancer/discussions/categories/ideas) discussion category.

### Reporting issues or bugs
If you find a bug in the code or want to request a new well-developed feature, please open an issue.

## NeuroMANCER development plan
Here are some upcoming features we plan to develop. Please let us know if you would like to get involved and 
contribute so we may be able to coordinate on development. If there is a feature that you think would
be highly valuable but not included below, please open an issue and let us know your thoughts. 

+ Control and modelling for networked systems
+ Support for stochastic differential equations (SDEs)
+ Easy to implement modeling and control with uncertainty quantification
+ Proximal operators for dealing with equality and inequality constraints
+ Interface with CVXPYlayers
+ Online learning examples
+ Benchmark examples of DPC compared to deep RL
+ Conda and pip package distribution
+ More versatile and simplified time series dataloading
+ Discovery of governing equations from learned RHS via NODEs and SINDy
+ More domain science examples


##  Release notes

### Version 1.4.1 Release Notes
+ To simplify integration, interpolation of control input is no longer supported in `integrators.py`
  + The `interp_u` parameter of `Integrator` and subclasses has been deprecated
+ Additional inputs (e.g., `u`, `t`) can now be passed as `*args` (instead of as a single tensor input stacked with `x`) in:
  + `Integrator` and subclasses in `integrators.py`
  + `Block` - new base class for all other classes in `blocks.py`
  + `ODESystem` in `ode.py`
+ New Physics-Informed Neural Network (PINN) examples for solving PDEs in `/examples/PDEs/`
+ New system identification examples for ODEs in `/examples/ODEs/`
+ Fixed a bug in the `show(...)` method of the `Problem` class
+ Hotfix: `*args` for `GeneralNetworkedODE`

###  Version 1.4 Release Notes
+ Refactored PSL
  + Better PSL unit testing coverage
  + Consistent interfaces across system types
  + Consistent perturbation signal interface in signals.py
+ Refactored Control and System ID learning using Node and System class (system.py)
  + Classes used for system ID can now be easily interchanged to accommodate downstream control policy learning

###  Version 1.3.2 Release Notes
+ Merged Structured Linear Maps and Pyton Systems Library into Neuromancer
  + The code in neuromancer was closely tied to psl and slim.
  A decision was made to integrate the packages as submodules of neuromancer.
  This also solves the issue of the package names "psl" and "slim" already being taken on PyPI.

*Import changes for psl and slim*

```python
# before
import psl
import slim

# now
from neuromancer import psl
from neuromancer import slim
```

### Version 1.3.1 release notes
+ New example scripts and notebooks
  + Interactive Colab notebooks for testing Neuromancer functionality without setting up an environment 
    + See [Examples](#examples) for links to Colab
  + RC-Network modeling using Graph Neural Time-steppers example:
    + See neuromancer/examples/graph_timesteppers/
  + Baseline NODE dynamics modeling results for all nonautonomous systems in Python Systems Library
    + See neuromancer/examples/benchmarks/node/
  + Updated install instructions for Linux, Windows, and MAC operating systems
    + New linux_env.yml, windows_env.yml, osxarm64_env.yml files for installation of dependencies across OS
+ Corresponding releases of SLiM and PSL packages
  + Make sure to update these packages if updating Neuromancer
  + Release 1.4 will roll SLiM and PSL into core Neuromancer for ease of installation and development

###  Version 1.3 release notes
+ Tutorial [YouTube videos](https://www.youtube.com/channel/UC5oWRFxzUwWrDNzkdWLIb7A) to accompany tutorial scripts in examples folder:
  + [examples/system_identification/duffing_parameter.py](https://www.youtube.com/watch?v=HLuqneSnoC8)
+ Closed loop control policy learning examples with Neural Ordinary Differential Equations
  + examples/control/
      + vdpo_DPC_cl_fixed_ref.py
      + two_tank_sysID_DPC_cl_var_ref.py
      + two_tank_DPC_cl_var_ref.py
      + two_tank_DPC_cl_fixed_ref.py
+ Closed loop control policy learning example with Linear State Space Models. 
  + examples/control/
      + double_integrator_dpc_ol_fixed_ref.py
      + vtol_dpc_ol_fixed_ref.py
+ New class for Linear State Space Models (LSSM)
    + LinearSSM in dynamics.py
+ Refactored closed-loop control policy simulations
  + simulator.py
+ Interfaces for open and closed loop simulation (evaluation after training) for several classes 
    + Dynamics
    + Estimator
    + Policy
    + Constraint
    + PSL Emulator classes
+ New class for closed-loop policy learning of non-autonomous ODE systems
  + ControlODE class in ode.py
+ Added support for NODE systems
  + Torchdiffeq integration with fast adjoint method for NODE optimization


## Development team

**Lead developers**: Aaron Tuor, Jan Drgona  
**Active PNNL developers**: James Koch, Madelyn Shapiro, Ethan King, Draguna Vrabie  
**Community contributors**: Seth Briney, Bo Tang  
**Past contributors**: Shrirang Abhyankar, Mia Skomski, Stefan Dernbach, Zhao Chen, Christian Møldrup Legaard


## Publications
+ [James Koch, Zhao Chen, Aaron Tuor, Jan Drgona, Draguna Vrabie, Structural Inference of Networked Dynamical Systems with Universal Differential Equations, arXiv:2207.04962, (2022)](https://aps.arxiv.org/abs/2207.04962)
+ [Ján Drgoňa, Sayak Mukherjee, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Learning Stochastic Parametric Differentiable Predictive Control Policies, IFAC ROCOND conference (2022)](https://www.sciencedirect.com/science/article/pii/S2405896322015877)
+ [Sayak Mukherjee, Ján Drgoňa, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Neural Lyapunov Differentiable Predictive Control, IEEE Conference on Decision and Control Conference 2022](https://arxiv.org/abs/2205.10728)
+ [Wenceslao Shaw Cortez, Jan Drgona, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Differentiable Predictive Control with Safety Guarantees: A Control Barrier Function Approach, IEEE Conference on Decision and Control Conference 2022](https://arxiv.org/abs/2208.02319)
+ [Ethan King, Jan Drgona, Aaron Tuor, Shrirang Abhyankar, Craig Bakker, Arnab Bhattacharya, Draguna Vrabie, Koopman-based Differentiable Predictive Control for the Dynamics-Aware Economic Dispatch Problem, 2022 American Control Conference (ACC)](https://ieeexplore.ieee.org/document/9867379)
+ [Drgoňa, J., Tuor, A. R., Chandan, V., & Vrabie, D. L., Physics-constrained deep learning of multi-zone building thermal dynamics. Energy and Buildings, 243, 110992, (2021)](https://www.sciencedirect.com/science/article/pii/S0378778821002760)
+ [E. Skomski, S. Vasisht, C. Wight, A. Tuor, J. Drgoňa and D. Vrabie, "Constrained Block Nonlinear Neural Dynamical Models," 2021 American Control Conference (ACC), 2021, pp. 3993-4000, doi: 10.23919/ACC50511.2021.9482930.](https://ieeexplore.ieee.org/document/9482930)
+ [Skomski, E., Drgoňa, J., & Tuor, A. (2021, May). Automating Discovery of Physics-Informed Neural State Space Models via Learning and Evolution. In Learning for Dynamics and Control (pp. 980-991). PMLR.](https://proceedings.mlr.press/v144/skomski21a.html)
+ [Drgoňa, J., Tuor, A., Skomski, E., Vasisht, S., & Vrabie, D. (2021). Deep Learning Explicit Differentiable Predictive Control Laws for Buildings. IFAC-PapersOnLine, 54(6), 14-19.](https://www.sciencedirect.com/science/article/pii/S2405896321012933)
+ [Tuor, A., Drgona, J., & Vrabie, D. (2020). Constrained neural ordinary differential equations with stability guarantees. arXiv preprint arXiv:2004.10883.](https://arxiv.org/abs/2004.10883)
+ [Drgona, Jan, et al. "Differentiable Predictive Control: An MPC Alternative for Unknown Nonlinear Systems using Constrained Deep Learning." Journal of Process Control Volume 116, August 2022, Pages 80-92](https://www.sciencedirect.com/science/article/pii/S0959152422000981)
+ [Drgona, J., Skomski, E., Vasisht, S., Tuor, A., & Vrabie, D. (2020). Dissipative Deep Neural Dynamical Systems, in IEEE Open Journal of Control Systems, vol. 1, pp. 100-112, 2022](https://ieeexplore.ieee.org/document/9809789)
+ [Drgona, J., Tuor, A., & Vrabie, D., Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees, arXiv preprint arXiv:2004.11184, (2020)](https://arxiv.org/abs/2004.11184)


## Cite as
```yaml
@article{Neuromancer2023,
  title={{NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations}},
  author={Tuor, Aaron and Drgona, Jan and Koch, James and Shapiro, Madelyn and Vrabie, Draguna and Briney, Seth},
  Url= {https://github.com/pnnl/neuromancer}, 
  year={2023}
}
```

## Acknowledgments
This research was partially supported by the Mathematics for Artificial Reasoning in Science (MARS) and Data Model Convergence (DMC) initiatives via the Laboratory Directed Research and Development (LDRD) investments at Pacific Northwest National Laboratory (PNNL), by the U.S. Department of Energy, through the Office of Advanced Scientific Computing Research's “Data-Driven Decision Control for Complex Systems (DnC2S)” project, and through the Energy Efficiency and Renewable Energy, Building Technologies Office under the “Dynamic decarbonization through autonomous physics-centric deep learning and optimization of building operations” and the “Advancing Market-Ready Building Energy Management by Cost-Effective Differentiable Predictive Control” projects. 
PNNL is a multi-program national laboratory operated for the U.S. Department of Energy (DOE) by Battelle Memorial Institute under Contract No. DE-AC05-76RL0-1830.

