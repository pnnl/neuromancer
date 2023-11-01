
##  Release notes

### Version 1.4.2 Release Notes 
+ Insert Here
  + Insert Here

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
