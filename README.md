# NeuroMANCER

![UML diagram](figs/class_diagram.png)

## Setup

##### Clone and install neuromancer, linear maps, and emulator packages
```console
user@machine:~$ mkdir neuromancer; cd neuromancer
user@machine:~$ git clone -b key_fix_cleanup https://stash.pnnl.gov/scm/deepmpc/deepmpc.git
user@machine:~$ git clone https://stash.pnnl.gov/scm/deepmpc/slip.git
user@machine:~$ git clone https://stash.pnnl.gov/scm/deepmpc/slip.git
```

##### Create the environment via .yml (Linux)

```console
user@machine:~$ conda env create -f env.yml
```

##### Create the environment via .yml (Windows)


```console
user@machine:~$ conda env create -f windows_env.yml
```

##### Create the environment manually

```console
user@machine:~$ conda config --add channels conda-forge

user@machine:~$ conda create -n neuromancer python=3.7
user@machine:~$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
user@machine:~$ conda install scipy pandas matplotlib control pyts numba scikit-learn mlflow dill
user@machine:~$ conda install -c powerai gym
```

### install neuromancer ecosystem 

```console
user@machine:~$ source activate neuromancer
(neuromancer) user@machine:~$ cd slip
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../slim
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../deepmpc
(neuromancer) user@machine:~$ python setup.py develop
```

