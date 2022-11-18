# NeuroMANCER v1.2
Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations.

The documentation for the project can be found in [documentation](https://pnnl.github.io/neuromancer/). 
Authors: Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen, Draguna Vrabie

![UML diagram](figs/class_diagram.png)

## Installation

First clone the neuromancer, slim, and psl libraries.

```bash

user@machine:~$ git clone -b master https://gitlab.pnnl.gov/dadaist/neuromancer.git --single-branch
user@machine:~$ git clone -b master https://gitlab.pnnl.gov/dadaist/psl.git --single-branch
user@machine:~$ git clone -b master https://gitlab.pnnl.gov/dadaist/slim.git --single-branch

```
## Install dependencies using .yml file

For Ubuntu users the simplest way to install NeuroMANCER dependencies is via the env.yml file.

``` bash
$ conda env create -f env.yml
$ conda activate neuromancer
(neuromancer) $ 
```

For MAC OS and Window users you may have to install dependencies manually.
conda install -c defaults intel-openmp -f
## Install dependencies using Conda
``` bash
$ conda create -n neuromancer python=3.10.4
$ conda activate neuromancer
(neuromancer) $ conda config --add channels conda-forge
(neuromancer) $ conda install pytorch cudatoolkit=10.2 -c pytorch
(neuromancer) $ conda install scipy numpy matplotlib scikit-learn pandas dill mlflow pydot=1.4.2 pyts numba networkx
(neuromancer) $ conda install networkx plum-dispatch
(neuromancer) $ conda install -c anaconda pytest hypothesis
```

## Install neuromancer ecosystem
```console
(neuromancer) user@machine:~$ cd ../psl
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../slim
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../neuromancer 
(neuromancer) user@machine:~$ python setup.py develop
```

## Torch-scatter
Torch-scatter requires a pip install which usually fails when using conda create with a yml file. 
If you get an error message about the pip install you will need to know the pytorch and 
cuda versions on your conda environment (these should be installed already if using the env.yml file).

``` bash
$ (neuromancer) python -c "import torch; print(torch.__version__)"
1.12.1
$ (neuromancer) python -c "import torch; print(torch.version.cuda)"
10.2
(neuromancer) $ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
```

## Examples

For detailed examples of NeuroMANCER usage
for control, system identification, and parametric programming as well as tutorials for basic usage, see the scripts
in the examples folder. 

The parametric programming examples have additional package dependencies for benchmarking
against traditional constrained optimization solvers, e.g., cvxpy (these should also have been installed using env.yml)

```console
(neuromancer) user@machine:~$ conda install cvxpy cvxopt casadi seaborn
```
## For developers
All test code is developed using pytest and hypothesis. Please refer to 
the test folder and create unit tests for any new modules introduced to the library. 

## Publications
+ James Koch, Zhao Chen, Aaron Tuor, Jan Drgona, Draguna Vrabie, Structural Inference of Networked Dynamical Systems with Universal Differential Equations, 	arXiv:2207.04962, (2022)
+ Ján Drgoňa, Sayak Mukherjee, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie, Learning Stochastic Parametric Differentiable Predictive Control Policies, arXiv:2205.10728, accepted for IFAC ROCOND conference (2022)
+ Sayak Mukherjee, Ján Drgoňa, Aaron Tuor, Mahantesh Halappanavar, Draguna Vrabie Neural Lyapunov Differentiable Predictive Control, arXiv:2205.10728, (2022)
+ Ethan King, Jan Drgona, Aaron Tuor, Shrirang Abhyankar, Craig Bakker, Arnab Bhattacharya, Draguna Vrabie, Koopman-based Differentiable Predictive Control for the Dynamics-Aware Economic Dispatch Problem, arXiv:2203.08984, 2022 American Control Conference (ACC), (2022) 
+ Drgoňa, J., Tuor, A. R., Chandan, V., & Vrabie, D. L., Physics-constrained deep learning of multi-zone building thermal dynamics. Energy and Buildings, 243, 110992, (2021)
+ E. Skomski, S. Vasisht, C. Wight, A. Tuor, J. Drgoňa and D. Vrabie, "Constrained Block Nonlinear Neural Dynamical Models," 2021 American Control Conference (ACC), 2021, pp. 3993-4000, doi: 10.23919/ACC50511.2021.9482930.
+ Skomski, E., Drgoňa, J., & Tuor, A. (2021, May). Automating Discovery of Physics-Informed Neural State Space Models via Learning and Evolution. In Learning for Dynamics and Control (pp. 980-991). PMLR.
+ Drgoňa, J., Tuor, A., Skomski, E., Vasisht, S., & Vrabie, D. (2021). Deep Learning Explicit Differentiable Predictive Control Laws for Buildings. IFAC-PapersOnLine, 54(6), 14-19.
+ Tuor, A., Drgona, J., & Vrabie, D. (2020). Constrained neural ordinary differential equations with stability guarantees. arXiv preprint arXiv:2004.10883.
+ Drgona, Jan, et al. "Differentiable Predictive Control: An MPC Alternative for Unknown Nonlinear Systems using Constrained Deep Learning." arXiv preprint arXiv:2011.03699 (2020).
+ Drgona, J., Skomski, E., Vasisht, S., Tuor, A., & Vrabie, D. (2020). Dissipative Deep Neural Dynamical Systems, arXiv preprint arXiv:2011.13492.
+ Drgona, J., Tuor, A., & Vrabie, D., Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees, arXiv preprint arXiv:2004.11184, (2020)

## Cite as
```yaml
@article{Neuromancer2022,
  title={{NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations}},
  author={Tuor, Aaron and Drgona, Jan and Skomski, Mia and Koch, James and Chen, Zhao and Dernbach, Stefan and Legaard, Christian Møldrup and Vrabie, Draguna},
  Url= {https://github.com/pnnl/neuromancer}, 
  year={2022}
}
```
