<p align="center">
  <img src="figs/Neuromancer.png" width="250">  
</p>

# Installation Instructions



## Table of Contents

1. [PIP Install (Recommended)](#pip-install-recommended)
2. [Manual Install](#manual-install)
   - [Clone the Repository](#clone-the-repository)
   - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
   - [Install NeuroMANCER and Dependencies](#install-neuromancer-and-dependencies)
   - [Notes for MacOS (Apple M1)](#notes-for-macos-apple-m1)
3. [Conda Install](#conda-install)
   - [Create Environment and Install Dependencies](#create-environment--install-dependencies)
     - [Ubuntu](#ubuntu)
     - [Windows](#windows)
     - [MacOS (Apple Silicon)](#macos-apple-silicon)
     - [Other (Manual Installation)](#other-manual-installation)
   - [Install NeuroMANCER Package](#install-neuromancer-package)
4. [Install Graphviz (Optional)](#install-graphviz-optional)
   - [Windows](#windows-1)
   - [Linux (Debian, Ubuntu)](#linux-debian-ubuntu)
   - [MacOS](#macos)
5. [Test NeuroMANCER Installation](#test-neuromancer-installation)

---

## PIP Install (Recommended)

Using `pip` is the simplest way to install NeuroMANCER. It's recommended to use a dedicated virtual environment (e.g., `conda`, `venv`) with Python 3.9+.


## Manual Install
### Clone the Repository
First clone the neuromancer package.
A dedicated virtual environment (conda or otherwise) is recommended. 

Note: If you have a previous neuromancer env it would be best at this point to create a new environment given the following instructions.

```bash
git clone -b master https://github.com/pnnl/neuromancer.git --single-branch
```

### Create and Activate Virtual Environment

``` bash
conda create -n neuromancer python=3.10.4
conda activate neuromancer
```

### Install Neuromancer and All Dependencies.
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

### Notes for MacOS (Apple M1)
Before CVXPY can be installed on Apple M1, you must install `cmake` via Homebrew:

```zsh
brew install cmake
```

See [CVXPY installation instructions](https://www.cvxpy.org/install/index.html) for more details.



## Conda install
Conda install is recommended for GPU acceleration. 

> ❗️Warning: `linux_env.yml`, `windows_env.yml`, and `osxarm64_env.yml` are out of date. Manual installation of dependencies is recommended for conda.


### Create Environment and Install Dependencies
#### Ubuntu

``` bash
conda env create -f linux_env.yml
conda activate neuromancer
```

#### Windows

``` bash
conda env create -f windows_env.yml
conda activate neuromancer
conda install -c defaults intel-openmp -f
```

#### MacOS (Apple Silicon)

``` bash
conda env create -f osxarm64_env.yml
conda activate neuromancer
```

#### Other (manually install all dependencies)

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

### Install NeuroMANCER package
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