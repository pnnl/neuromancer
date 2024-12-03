
<p align="center">
  <img src="figs/Neuromancer.png" width="250">  
</p>

# NeuroMANCER v1.5.1

**Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations (NeuroMANCER)**
is an open-source differentiable programming (DP) library for solving parametric constrained optimization problems, 
physics-informed system identification, and parametric model-based optimal control.
NeuroMANCER is written in [PyTorch](https://pytorch.org/) and allows for systematic 
integration of machine learning with scientific computing for creating end-to-end 
differentiable models and algorithms embedded with prior knowledge and physics.

### ⭐ Now available on PyPi! ⭐
![Static Badge](https://img.shields.io/badge/pip_install-neuromancer-blue)
 ![PyPI - Version](https://img.shields.io/pypi/v/neuromancer)


### New in v1.5.1
#### Lightning Enhancements 
![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)
Further integration enhancements with PyTorch Lightning (https://lightning.ai/docs/pytorch/stable/). We now support all Lightning hooks, which are modular logic blocks that make defining custom training/validation/etc logic very easy and user-friendly.

We have also released a *Lightning Studio* course on **Differentiable Predictive Control**: <a target="_blank" href="https://lightning.ai/rahulbirmiwal/studios/differential-predictive-control-with-neuromancer">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio" style="width:100px; height:auto;"/>
</a>!

Lightning Studios are powerful, AI development platforms. They essentially act as extremely user-friendly virtual machines, but accessible through your browser! Please see https://lightning.ai/studios for more information. 


#### TorchSDE Integration
* We have begun integration with the TorchSDE library (https://github.com/google-research/torchsde/tree/master) TorchSDE provides stochastic differential equation solvers with GPU spport and efficient backpropagation. 
* Neuromancer already has robust and extensive library for Neural ODEs and ODE solvers. We extend that functionality to the stochastic case by incorporating TorchSDE solvers. To motivate and teach the user how one progresses from neural ODEs to "neural SDEs" we have written a lengthy notebook -- [sde_walkthrough.ipynb](examples/SDEs/sde_walkthrough.ipynb)

#### Stacked Physics-Informed Neural Networks
* Neuromancer now supports Stacked Physics-Informed Neural Networks. This architecture, based on the work of [Howard et al. (2023)](https://arxiv.org/abs/2311.06483), consists of stacking multifidelity networks via composition, allowing a progressive improvement of learned solutions. This formulation is especially useful for highly oscillatory problems. We illustrate an example of its usage with the solution of a damped harmonic oscillator using PINN: [Part_5_Pendulum_Stacked.ipynb](examples/PDEs/Part_5_Pendulum_Stacked.ipynb)

#### SINDy

* Sparse Identification of Nonlinear Dynamics (SINDy) is a powerful method which uses sparse regression to identify a small number of active terms in dynamic systems, allowing for interpretable and efficient modeling of complex, nonlinear dynamics. We now enable users to leverage this technique for sparse physics-informed system identification. Checkout the notebook here [Part_9_SINDy.ipynb](examples/ODEs/Part_9_SINDy.ipynb)

**New Colab Examples:**
> ⭐ [Custom Training Via Lightning Hooks ](#lightning-integration-examples)

> ⭐ [Latent Stochastic Differential Equations ](#stochastic-differential-equation-examples)

> ⭐ [Stacked Physics-Informed Neural Networks ](examples/PDEs/Part_5_Pendulum_Stacked.ipynb)

> ⭐ [Part 9: Sparse Identification of Nonlinear Dynamics (SINDy) ](#ordinary-differential-equation-examples)


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


### Learning to Optimize (L2O) for Parametric Programming

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_1_basics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 1: Learning to solve a constrained optimization problem.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_2_pQP.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 2: Learning to solve a quadratically-constrained optimization problem.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_3_pNLP.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Part 3: Learning to solve a set of 2D constrained optimization problems.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_4_projectedGradient.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 
Part 4: Learning to solve a constrained optimization problem with the projected gradient.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_5_cvxpy_layers.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 
Part 5: Using Cvxpylayers for differentiable projection onto the polytopic feasible set.  

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_6_pQp_lopoCorrection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 
Part 6: Learning to optimize with metric learning for Operator Splitting layers.  

### System Identification of Ordinary Differential Equations (ODEs)
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

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_9_SINDy.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 9: Sparse Identification of Nonlinear Dynamics (SINDy)


### Physics-Informed Neural Networks (PINNs) for Partial Differential Equations (PDEs) and Ordinary Differential Equations (ODEs)
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_1_PINN_DiffusionEquation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Diffusion Equation
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_2_PINN_BurgersEquation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Burgers' Equation
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_3_PINN_BurgersEquation_inverse.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Burgers' Equation w/ Parameter Estimation (Inverse Problem)
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_4_PINN_LaplaceEquationSteadyState.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: Laplace's Equation (steady-state)
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_5_Pendulum_Stacked.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 5: Damped Pendulum (stacked PINN)
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_6_PINN_NavierStokesCavitySteady_KAN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 6: Navier-Stokes equation (lid-driven cavity flow, steady-state, KAN)

### Learning to Control (L2C) with Differentiable Models

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

### Domain Examples 

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/DPC_building_control.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Learning to Control Indoor Air Temperature in Buildings.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/DPC_PSH.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Learning to Control an Pumped-Hydroelectricity Energy Storage System.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/NODE_building_dynamics.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Learning Building Thermal Dynamics using Neural ODEs.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/NODE_RC_networks.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: Data-driven modeling of a Resistance-Capacitance network with Neural ODEs.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/NODE_swing_equation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 5: Learning Swing Equation Dynamics using Neural ODEs.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/domain_examples/NSSM_building_dynamics.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 6: Learning Building Thermal Dynamics using Neural State Space Models.

### Lightning Integration Examples 

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/lightning_integration_examples/Part_1_lightning_basics_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Lightning Integration Basics.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/lightning_integration_examples/Part_2_lightning_advanced_and_gpu_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Lightning Advanced Features and Automatic GPU Support.

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/lightning_integration_examples/Part_4_lightning_wanb_hyperparameter_tuning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Hyperparameter Tuning With Lightning & WandB

+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/lightning_integration_examples/other_examples/lightning_custom_training_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 4: Defining Custom Training Logic via Lightning Modularized Code.
 

### Stochastic Differential Equations (SDEs) 
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/SDEs/sde_walkthrough.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> LatentSDEs: "System Identification" of Stochastic Processes using Neuromancer x TorchSDE


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

### PIP Install (recommended)

Consider using a dedicated virtual environment (conda or otherwise) with Python 3.9+ installed. 

```bash
pip install neuromancer
```
Example usage: 

```bash
import torch
from neuromancer.system import Node

fun_1 = lambda x1, x2: 2.*x1 - x2**2
node_3 = Node(fun_1, ['y1', 'y2'], ['y3'], name='quadratic')
# evaluate forward pass of the node with dictionary input dataset
print(node_3({'y1': torch.rand(2), 'y2': torch.rand(2)}))

```
### Manual Install

First clone the neuromancer package.
A dedicated virtual environment (conda or otherwise) is recommended. 

Note: If you have a previous neuromancer env it would be best at this point to create a new environment given the following instructions.

```bash
git clone -b master https://github.com/pnnl/neuromancer.git --single-branch
```

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
examples = ["casadi", "cvxpy", "imageio", "cvxpylayers"]
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

> ❗️Warning: `linux_env.yml`, `windows_env.yml`, and `osxarm64_env.yml` are out of date. Manual installation of dependencies is recommended for conda.


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

##### MacOS (Apple Silicon)

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
conda install lightning wandb -c conda-forge
## (for Windows): conda install -c defaults intel-openmp -f
```

#### Install NeuroMANCER package
From the top level directory of cloned neuromancer
(in the activated environment where the dependencies have been installed):

```bash
pip install -e . --no-deps
```

### Install Graphviz (optional)
In order to use the Problem graph plots, we recommend installing Graphviz system-wide. Note that this feature is optional.

#### For Windows:
Package must be installed manually: [Graphviz website](https://graphviz.org/download/)

#### For Linux (Debian, Ubuntu)
```bash
sudo apt install graphviz
```
#### For MacOS
```bash
brew install graphviz
```

### Test NeuroMANCER install
Run pytest on the [tests folder](https://github.com/pnnl/neuromancer/tree/master/tests). 
It should take about 2 minutes to run the tests on CPU. 
There will be a lot of warnings that you can safely ignore. These warnings will be cleaned 
up in a future release.

## Community Information
We welcome contributions and feedback from the open-source community!  

### Contributions, Discussions, and Issues
Please read the [Community Development Guidelines](https://github.com/pnnl/neuromancer/blob/master/CONTRIBUTING.md) 
for further information on contributions, [discussions](https://github.com/pnnl/neuromancer/discussions), and [Issues](https://github.com/pnnl/neuromancer/issues).

###  Release notes
See the [Release notes](https://github.com/pnnl/neuromancer/blob/master/RELEASE_NOTES.md) documenting new features.

###  License
NeuroMANCER comes with [BSD license](https://en.wikipedia.org/wiki/BSD_licenses).
See the [license](https://github.com/pnnl/neuromancer/blob/master/LICENSE.md) for further details. 


## Publications 
+ [Jan Drgona, Aaron Tuor, Draguna Vrabie, Learning Constrained Parametric Differentiable Predictive Control Policies With Guarantees, IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024](https://ieeexplore.ieee.org/abstract/document/10479163)
+ [Renukanandan Tumu, Wenceslao Shaw Cortez, Ján Drgoňa, Draguna L. Vrabie, Sonja Glavaski, Differentiable Predictive Control for Large-Scale Urban Road Networks, 	arXiv:2406.10433, 2024](https://arxiv.org/abs/2406.10433)
+ [Ethan King, James Kotary, Ferdinando Fioretto, Jan Drgona, Metric Learning to Accelerate Convergence of Operator Splitting Methods for Differentiable Parametric Programming, arXiv:2404.00882, 2024](https://arxiv.org/abs/2404.00882)
+ [James Koch, Madelyn Shapiro, Himanshu Sharma, Draguna Vrabie, Jan Drgona, Neural Differential Algebraic Equations, arXiv:2403.12938, 2024](https://arxiv.org/abs/2403.12938)
+ [Wenceslao Shaw Cortez, Jan Drgona, Draguna Vrabie, Mahantesh Halappanavar, A Robust, Efficient Predictive Safety Filter, arXiv:2311.08496, 2024](https://arxiv.org/abs/2311.08496)
+ [Shrirang Abhyankar, Jan Drgona, Andrew August, Elliott Skomski, Aaron Tuor, Neuro-physical dynamic load modeling using differentiable parametric optimization, 2023 IEEE Power & Energy Society General Meeting (PESGM), 2023](https://ieeexplore.ieee.org/abstract/document/10253098)
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
  author={Drgona, Jan and Tuor, Aaron and Koch, James and Shapiro, Madelyn and Jacob, Bruno and Vrabie, Draguna},
  Url= {https://github.com/pnnl/neuromancer}, 
  year={2023}
}
```

## Development team

**Active core developers**: [Jan Drgona](https://drgona.github.io/), [Rahul Birmiwal](https://www.linkedin.com/in/rahul-birmiwal009/), [Bruno Jacob](https://brunopjacob.github.io/)  
**Notable contributors**: [Aaron Tuor](https://sw.cs.wwu.edu/~tuora/aarontuor/), Madelyn Shapiro, James Koch, Seth Briney, Bo Tang, Ethan King, Elliot Skomski, Zhao Chen, Christian Møldrup Legaard  
**Scientific advisors**: Draguna Vrabie, Panos Stinis  

Open-source contributions made by:  
<a href="https://github.com/pnnl/neuromancer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnl/neuromancer" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Acknowledgments
This research was partially supported by the Mathematics for Artificial Reasoning in Science (MARS) and Data Model Convergence (DMC) initiatives via the Laboratory Directed Research and Development (LDRD) investments at Pacific Northwest National Laboratory (PNNL), by the U.S. Department of Energy, through the Office of Advanced Scientific Computing Research's “Data-Driven Decision Control for Complex Systems (DnC2S)” project, and through the Energy Efficiency and Renewable Energy, Building Technologies Office under the “Dynamic decarbonization through autonomous physics-centric deep learning and optimization of building operations” and the “Advancing Market-Ready Building Energy Management by Cost-Effective Differentiable Predictive Control” projects. 
PNNL is a multi-program national laboratory operated for the U.S. Department of Energy (DOE) by Battelle Memorial Institute under Contract No. DE-AC05-76RL0-1830.

