
##  Release notes

### Version 1.5.3 Release Notes
+ New feature: NeuroMANCER LLM Assistant configuration scripts
+ New feature: In-depth and educational notebook comparing and contrasting RL vs DPC for building energy systems control
+ New feature: Updated NODE class that can now accept instantiated Variables in the constructor

### Version 1.5.2 Release Notes
+ New feature: Multi-fidelty Kolgomorov Arnold Networks for SOTA function approximation
+ New feature: Load forecasting on building energy systems tutorials
+ New feature: Transformer block 


### Version 1.5.1 Release Notes
+ Enhancement: Now supports integration of all Lightning hooks into the Neuromancer Lightning trainer. Please refer to Lightning examples README for more information
+ Deprecated WandB hyperparameter tuning via `LitTrainer` for now 
+ New feature: TorchSDE integration with Neuromancer core library, namely `torchsde.sdeint()`. Motivating example for system ID on stochastic process found in examples/sdes/sde_walkthrough.ipynb
+ New feature: Stacked physics-informed neural networks 
+ New feature: SINDy -- sparse system identification of nonlinear dynamical systems
+ New feature: differentiable proximal operators in operator splitting methods for learning to optimize

### Version 1.5.0 Release Notes 
+ New Feature: PyTorch Lightning Integration with NeuroMANCER core library. All these features are opt-in. 
  + Code simplifications: zero boilerplate code, increased modularity 
  + Added ability for user to define custom training logic 
  + Easy support for GPU and multi-GPU training
  + Easy Weights and Biases (https://wandb.ai/site) hyperparameter tuning and Tensorboard Logging


### Version 1.4.2 Release Notes 
+ New feature: Update violation energy for projected gradient #110 (based on idea #86).
+ Reverted `psl.nonautonomous.TwoTank` `(umin, umax)` bounds to `(0.5, 0.5)` for numerical stability #105
+ Added new unit tests for `problem.py` and `system.py` #107
+ Automated docs build from `master -> gh-pages` #107
+ Fixed positional arg error and added support for Time data in `file_emulator.py` #119
+ Fixed a bug in `System` which caused incorrect visualization of the computational graph
+ New examples:
  + [learning to optimize with cvxplayers](https://github.com/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_5_cvxpy_layers.ipynb)
  + [Deep Koopman for system identification](https://github.com/pnnl/neuromancer/blob/master/examples/ODEs/Part_7_DeepKoopman.ipynb)
  + [control-oriented Deep Koopman for system identification](https://github.com/pnnl/neuromancer/blob/master/examples/ODEs/Part_8_nonauto_DeepKoopman.ipynb)
  + extended set of [domain examples](https://github.com/pnnl/neuromancer/tree/master/examples/domain_examples)
+ Other minor updates to examples

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