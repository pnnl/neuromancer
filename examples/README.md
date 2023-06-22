# NeuroMANCER code examples

## Files

Here we list the file tree and plots of expected results from the default parameters of the example scripts

+ **Neuromancer tutorials**/ Introducing basic functionality of the library.
    - component_tutorial.py: Example code of working with the Neuromancer Component class          
    - constraints_tutorial.py: Example code for easy formulation of objectives and constraints usint the Neuromancer Variable and Constraint classes.  
    - graph_timestepper_tutorial.py: Example code of system identification using the Neuromancer graph timestepper.   
    - linear_regression.py: Example code using new Neuromancer Variable class to solve a simple linear regression problem
    - component_tutorial_interactive.py: Example code of working with the Neuromancer Component class 
    - dataset_tutorial.py: Example code of using Neuromancer built in data loading code using pytorch Datasets and Dataloaders
    - integrators_vs_psl_auto.py: Comparison of neuromancer numerical integration implementation with PSL simulations using scipy's odeint solver
    - toy_interpolation.py: Use of control action interpolation with neuromancer neural ODEs


+ **parametric_programming**/ Examples for solving a set of parametric programming problems.
  + <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Himmelblau_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[Part 1](./Part_1_basics.py):
Formulating and solving parametric nonlinear programming (pNLP) problem.  
  +   <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Himmelblau_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[Part 2](./Part_2_pQP.py): Formulating and solving two parametric quadratic programming (pQP) problems.    
  +   <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Himmelblau_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[Part 3](./Part_3_pNLP.py): Formulating and solving a set of parametric nonlinear programming (pNLP) problems.  


+ **system_identification/** This folder has examples for system identification of autonomous and nonautonomous dynamical systems.
    - **boxes.pkl**: Pickle file with reasonable bounds for initial conditions for the chaotic systems. 
    - **brusselator_parameter.py**: Parameter estimation for a 1D Brusselator system.
        + ![brusselator](figs/brusselator_parameter.png)
    - **lorenz_control_node_curriculum.py**: Modeling Lorenz system 
      with additional stabilizing additive control using neural ODEs and curriculum learning. 
        + ![](figs/lorenz_control_node_curriculum.png)
    - **lorenz_node_curriculum.py**: Modeling Lorenz system 
       using neural ODEs and curriculum learning. 
       + ![](figs/lorenz_node_curriculum.png)
    - **duffing_parameter.py**: Parameter estimation of Duffing ODE system. <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/system_identification/duffing_parameter_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        + ![duffing](figs/duffing_parameter.png)
    - **two_tank_node.py**: Modeling a Two Tank system with neural ODEs
        + ![two_tank](figs/two_tank_neural_ode.png)
    - **two_tank_neural_ssm.py**: Modeling a two tank system with a neural state space model.
        + ![cstr](figs/two_tank_neural_ssm.png)

+ **control**/ Examples for solving a set of model-based optimal control problems.
    + double_integrator_dpc.py: Stabilizing control for a double integrator system. <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/control/double_integrator_DPC_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+ **graph_timesteppers**/ Scripts for various graph timestepper models used for system identification.
    + rc_network.py: Resistor-Capacitor Network simulation for heat flow in a 5 room house.
        - ![](figs/rcnetwork.png)

+ **figs**/ Plots of expected results for the respective scripts that generated them.
