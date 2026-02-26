# TorchSDE x NeuroMANCER

The example in this folder, sde_walkthrough.ipynb, demonstrates how functionality from TorchSDE can be, and is, integrated into the Neuromancer workflow. https://github.com/google-research/torchsde/tree/master

TorchSDE provides stochastic differential equation solvers with GPU spport and efficient backpropagation. They are based off this paper: http://proceedings.mlr.press/v108/li20i.html

Neuromancer already has robust and extensive library for Neural ODEs and ODE solvers. We extend that functionality to the stochastic case by incorporating TorchSDE solvers. To motivate and teach the user how one progresses from neural ODEs to "neural SDEs" we have written a lengthy notebook -- sde_walkthrough.ipynb

Please ensure torchsde is installed: 
```
pip install torchsde
```
