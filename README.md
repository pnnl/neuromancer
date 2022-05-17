# NeuroMANCER
## Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations

## [Complete Documentation](https://pnnl.github.io/neuromancer/)
![UML diagram](figs/class_diagram.png)

## Setup

##### Clone and install neuromancer, linear maps, and emulator submodules 
```console

user@machine:~$ git clone https://gitlab.pnnl.gov/dadaist/neuromancer.git
user@machine:~$ git clone https://gitlab.pnnl.gov/dadaist/psl.git
user@machine:~$ git clone https://gitlab.pnnl.gov/dadaist/slim.git

```

##### Create the environment via .yml (Linux)

```console
user@machine:~$ conda env create -f env.yml
(neuromancer) user@machine:~$ source activate neuromancer
```

##### If .yml env creation fails create the environment manually

```console
user@machine:~$ conda config --add channels conda-forge pytorch
user@machine:~$ conda create -n neuromancer python=3.7
user@machine:~$ source activate neuromancer
(neuromancer) user@machine:~$ conda install pytorch torchvision -c pytorch
(neuromancer) user@machine:~$ conda install scipy pandas matplotlib control pyts numba scikit-learn dill
(neuromancer) user@machine:~$ conda install mlflow boto3
(neuromancer) user@machine:~$ conda install -c powerai gym
(neuromancer) user@machine:~$ conda install -c anaconda pytest
(neuromancer) user@machine:~$conda install hypothesis
```

##### Install neuromancer ecosystem 


```console
(neuromancer) user@machine:~$ cd ../psl
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../slim
(neuromancer) user@machine:~$ python setup.py develop
(neuromancer) user@machine:~$ cd ../neuromancer 
(neuromancer) user@machine:~$ python setup.py develop
```

### TODOs

datasets category
    [ ] In datasets add data_transforms method to act on dataset.data to generate: finite difference sequences via np.diff, nonlinear expansions, spectral decompositions


blocks, dynamics, policies, estimators category    
    [ ] Re-implement RNN state preservation for open loop simulation
    [ ] full trajectory estimators: This will entail only taking the first N-steps for all the non-static inputs
    [ ] Pytorch Extended Kalman Filter: 
            https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/EKF.html
    [ ] Implement LQR policy, similar structure to Linear Kalman Filter: 
            Scipy reference https://nbviewer.jupyter.org/url/argmin.net/code/little_LQR_demo.ipynb