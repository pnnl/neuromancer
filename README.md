# NeuroMANCER v1.3
Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations.

Authors: Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen,
Christian Møldrup Legaard, Draguna Vrabie

## Version 1.3 release notes

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
+ Refactored closed loop control policy simulations
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


## Documentation
The documentation for the library can be found [online](https://pnnl.github.io/neuromancer/)
and in the [pdf form](https://github.com/pnnl/neuromancer/blob/master/Documentation.pdf). 
There is also an [introduction video](https://www.youtube.com/watch?v=YkFKz-DgC98) covering 
core features of the library. 


```python 
# Neuromancer syntax example for differentiable parametric programming
import neuromancer as nm

# primal solution map to be trained
func = nm.blocks.MLP(insize=2, outsize=2, hsizes=[80] * 4)
sol_map = nm.maps.Map(func,
        input_keys=["a", "p"],
        output_keys=["x"],
        name='primal_map')

# problem primal variables
x = nm.constraints.variable("x")[:, [0]]
y = nm.constraints.variable("x")[:, [1]]

# sampled problem parameters
p = nm.constraints.variable('p')
a = nm.constraints.variable('a')

# nonlinear objective function
f = (1-x)**2 + a*(y-x**2)**2
obj = f.minimize(weight=1., name='obj')

# constraints
con_1 = 100*(x >= y)
con_2 = 100*((p/2)**2 <= x**2+y**2)
con_3 = 100*(x**2+y**2 <= p**2)

# create constrained optimization loss
objectives = [obj]
constraints = [con_1, con_2, con_3]
loss = nm.loss.PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
components = [sol_map]
problem = nm.problem.Problem(components, loss)
```

![UML diagram](figs/class_diagram.png)
*UML diagram of NeuroMANCER classes.*


## Installation

First clone the neuromancer, [slim](https://github.com/pnnl/slim), and [psl](https://github.com/pnnl/psl) libraries.

```bash

user@machine:~$ git clone -b master https://github.com/pnnl/neuromancer.git --single-branch
user@machine:~$ git clone -b master https://github.com/pnnl/psl.git --single-branch
user@machine:~$ git clone -b master https://github.com/pnnl/slim.git --single-branch

```

## Install neuromancer ecosystem
``` bash
(neuromancer) $ cd psl; python setup.py develop
(neuromancer) $ cd ../slim; python setup.py develop
(neuromancer) $ cd ../neuromancer; python setup.py develop
``` 

## Install dependencies

### Ubuntu

``` bash
$ conda env create -f env.yml
$ conda activate neuromancer
```

### Windows

``` bash
$ conda env create -f windows_env.yml
$ conda activate neuromancer
(neuromancer) $ conda install -c defaults intel-openmp -f
```

### Other operating system

``` bash
$ conda create -n neuromancer python=3.10.4
$ conda activate neuromancer
(neuromancer) $ conda config --add channels conda-forge
(neuromancer) $ conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
(neuromancer) $ conda install scipy numpy matplotlib scikit-learn pandas dill mlflow pydot=1.4.2 pyts numba
(neuromancer) $ conda install networkx plum-dispatch 
(neuromancer) $ conda install -c anaconda pytest hypothesis
(neuromancer) $ conda install cvxpy cvxopt casadi seaborn
(neuromancer) $ conda install tqdm
(neuromancer) $ conda install torchdiffeq
```

## Test NeuroMANCER install
Run pytest on the test folder. It should take about 2 minutes to run the tests on CPU. 
There will be a lot of warnings that you can safely ignore. These warnings will be cleaned 
up in a future release.

## Examples

For detailed examples of NeuroMANCER usage
for control, system identification, and parametric programming as well as tutorials for basic usage, see the scripts
in the examples folder. 

## Community

### Contributing examples
If you have an example of using NeuroMANCER to solve an interesting problem, or of using 
NeuroMANCER in a unique way, we would love to see it incorporated into our current library of examples. 
To submit an example, create a folder for your example/s in the example folder if there isn't 
currently and applicable folder and place either your executable python file or notebook file there. 
Push your code back to github and then submit a pull request. Please make sure to note in a comment at 
the top of your code if there are additional dependencies to run your example and how to install 
those dependencies. 

### Contributing code

We welcome contributions to NeuroMANCER. Please accompany contributions with some lightweight unit tests
via pytest (see test/ folder for some examples of easy to compose unit tests using pytest). In addition to unit tests
a script utilizing introduced new classes or modules should be placed in the examples folder. 
To contribute a new feature please submit a pull request. 

### Reporting issues or bugs
If you find a bug in the code or want to request a new feature, please open an issue.

### NeuroMANCER development plan
Here are some upcoming features we plan to develop. Please let us know if you would like to get involved and 
contribute so we may be able to coordinate on development. If there is a feature that you think would
be highly valuable but not included below, please open an issue and let us know your thoughts. 

+ Faster dynamics modeling via Torchscript
+ Control and modelling for networked systems
+ Easy to implement modeling and control with uncertainty quantification
+ Online learning examples
+ Benchmark examples of DPC compared to deep RL
+ Conda and pip package distribution
+ CVXPY-like interface for optimization via Problem.solve method
+ More versatile and simplified time series dataloading
+ Pytorch Lightning trainer compatibility

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
@article{Neuromancer2022,
  title={{NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations}},
  author={Tuor, Aaron and Drgona, Jan and Skomski, Mia and Koch, James and Chen, Zhao and Dernbach, Stefan and Legaard, Christian Møldrup and Vrabie, Draguna},
  Url= {https://github.com/pnnl/neuromancer}, 
  year={2022}
}
```