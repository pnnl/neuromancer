<p align="center">
  <img src="figs/Neuromancer.png" width="250">  
</p>

# NeuroMANCER User and Developer Guide 

## Installation

For development purposes, we recommend setting up a dedicated virtual environment. 
```
# Create a virtual environment using Conda
conda create -n neuromancer-dev python=3.11
conda activate neuromancer-dev
git clone -b master https://github.com/pnnl/neuromancer.git --single-branch
cd neuromancer

# Install dependencies
pip install -e .[docs,tests,examples]
```

For detailed manual installation instructions please refer to [Installation Insturctions](https://github.com/pnnl/neuromancer/blob/master/INSTALLATION.md). 

## Code Structure
### Project Layout
```
neuromancer/
│
├── docs/                   # Documentation files
├── examples/               # Tutorials and example notebooks
├── src/                    # Source code for the NeuroMANCER library
│   ├── modules/            # Classes,functions to support our neural blocks API
│   ├── dynamics/           # ODE, SDE dynamics classes and integrators
│   ├── psl/                # Physics simulation library for data generation of variety of ODEs
│   ├── slim/               # structured linear maps which can be used as drop-in replacements for PyTorch’s nn.Linear module
│   ├── slim/               # structured linear maps which can be used as drop-in replacements for 
│   └── constraint.py       # Constraint and Variable classes    
│   ├── dataset.py          # Various Neuromancer API datasets e.g. DictDataset and LightningDataModule dataset
│   ├── system.py           # Node, System classes
│   ├── problem.py          # Problem and LitProblem (PyTorch Lightning verion) classes
│   ├── trainer.py          # Two Neuromancer trainers -- Trainer and LitTrainer (PyTorch Lightning version)
│   ├── ...                 # More classes
├── tests/                  # Unit tests and integration tests
├── INSTALLATION.md         # Detailed installation instructions
├── DEVELOPER.md            # Guidelines for developers and open-source contributors
├── LICENSE.md              # License information
└── README.md               # Main README for users
```


## Design Patterns and Import Modules  
### Key Design Patterns
NeuroMANCER uses several design patterns to provide flexibility and modularity:

* **Node and System Pattern**: Core abstractions where nodes represent computational elements, and systems represent composed collections of nodes.
* **Symbolic Programming**: Allows the definition of constraints, objectives, and optimization problems using symbolic representations and interfaces with our Node class. 
* **Modular Architecture**: Each module (e.g., neural network (block), loss, constraints) is modular and can be easily extended or replaced

### Import Modules
Using NeuroMANCER often involves two main components: the data and the "problem" one is trying to solve. 

### Data
We provide the user a variety of entry points into forward simulation of dynamical systems through the classes of PSL and Dynamics/ode.py:

ODESystem is a flexible base for building custom ODE models and doing grey-box, physics-informed neural ODEs. PSL provides ready-to-use, physics-based models to generate data or benchmark algorithms, offering less flexibility but faster setup for canonical problems, especially in controls domain. 

However the user may chooose to ingest data in anyway, as long as data is converted to DictDataset API (if necessary, e.g. in the case of L2M/L2C). 

### The Problem: What Are We Trying To Solve?
Neuromancer is designed to handle three types of problems (our Problem class is aptly named to serve as a wrapper for all three) -- learning to optimize (L2O), learning to model (L2M), and learning to control (L2C) paradigms. 

* **L2O**: Please refer to our parametric programming examples. For these types of problems, the user will be interfacing just with classes found in Modules, namely Node, Variable, and constructing constraints through the composition of these symbolic variables. 
* **L2M**: Please refer to the majority of our examples such as ODEs, PDEs, KANs. For these types of problems, the user will be using neural blocks to model the dynamics of the data chosen above. We provide a variety of blocks (see modules/blocks.py) and methods -- neural state space models, neural ODEs, neural SDEs, PINNs, KANs, etc -- for the user. 
* **L2C**: Please refer to our control examples and domain examples. Here the user will be working with the same set of neural blocks and methods as in the L2M case, but will attach a neural control block in closed-loop with the system identification model performed in the L2M step. Furthermore, the user will often incorporate constraints akin to those in the L2O case. For control problems, NeuroMANCER is best-suited for novel control methodology called **Differential Predictive Control**



All of our examples can be boiled down to some novel combination of data + problem formulation. To place greater emphasis on this design methodology, as well as provide a host of other software development/MLOps benefits, we have integrated our library with PyTorch Lightning. 

### UML Diagram

To illustrate the library we have the following UML diagram
![UML diagram](figs/class_diagram.png)
*UML diagram of NeuroMANCER classes.*



## PyTorch Lightning Integration

PyTorch Lightning is a framework built on top of PyTorch designed to simplify the implementation of complex models in PyTorch. It promotes cleaner, more organized code by providing a high-level interface for PyTorch that handles many training intricacies automatically. 

Why might I want to use Lightning in the context of NeuroMANCER? For the user, Lightning simplifies the workflow, allowing one to focus on solving NeuroMANCER problems with their associated datasets. How does it do this? Let's take a look at the added functionality when using Lightning. 

#### Lightning Features

* PyTorch boilerplate code is removed, thus increasing the accessibility of the NeuroMANCER library. For example, the user doesn't need to deal with details instantiating a PyTorch Dataloader() class. 
* Increased modularity on the data portion: Lightning caters towards modularity often found in "traditional" scientific experimental design. In the Lightning workflow, the user is expected to define a "data_setup_function" that is fed into the LightningTrainer(). The user can easily swap out, or modify generation of, the datasets from run-to-run. 
* Automatic GPU support: Lightning allows for easy migration to running training on the GPU. It even allows for multi-GPU training with a simple keyword argument change. Most of all, .to(device) calls are no longer required. 
* Other features include hyperparameter tuning with wandb, automatic logging, easy profiling integration, and ability for the user to define own training logic without boilerplate

**We strongly encourage all future contributions to leverage the PyTorch Lightning integration and ecosystem (e.g. using LitTrainer)**

## Development Workflow
### Git Workflow
We follow a standard Git Flow for managing development:

1. Fork the repository.
2. Clone your fork and create a new branch for your feature or bug fix.
```
git checkout -b feature/your-feature-name
```
3. Make your changes, commit them, and push to your fork
```
git commit -m "Add a description of your changes"
git push origin feature/your-feature-name
```

### Code Style Guide  
* Follow PEP 8 guidelines for Python code style.
* Maintain docstrings for all functions and classes


## Contributing
### How to Submit a PR
1. Ensure your changes follow the coding standards.
2. Write or update tests to cover your changes.
3. Document any new features in the code and/or documentation files.
4. Submit your Pull Request and request a review from the core maintainers.

### Notebook Development
Any methods development (such as a novel neural network architecture to be added to blocks.py) or domain example (such as traffic flow modeling and control) should be accompanied by an education tutorial notebook where one introduces the context of the problem, underlying mathematical equations, objective function (if applicable), NeuroMANCER problem formulation and results section. 

## Community Information
We welcome contributions and feedback from the open-source community!  

### Contributions, Discussions, and Issues
Please read the [Community Development Guidelines](https://github.com/pnnl/neuromancer/blob/master/CONTRIBUTING.md) 
for further information on contributions, [discussions](https://github.com/pnnl/neuromancer/discussions), and [Issues](https://github.com/pnnl/neuromancer/issues).

###  Release notes
See the [Release notes](https://github.com/pnnl/neuromancer/blob/master/RELEASE_NOTES.md) documenting new features.