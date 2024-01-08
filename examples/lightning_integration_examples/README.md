## 

## Motivation for PyTorch Lightning 

PyTorch Lightning is a framework built on top of PyTorch designed to simplify the implementation of complex models in PyTorch. It promotes cleaner, more organized code by providing a high-level interface for PyTorch that handles many training intricacies automatically. 

Why might I want to use Lightning in the context of NeuroMANCER? For the user, Lightning simplifies the workflow, allowing one to focus on solving NeuroMANCER problems with their associated datasets. How does it do this? Let's take a look at the added functionality when using Lightning 

## New Features

* PyTorch boilerplate code is removed, thus increasing the accessibility of the NeuroMANCER library. For example, the user doesn't need to deal with details instantiating a PyTorch Dataloader() class. 
* Increased modularity on the data portion: Lightning caters towards modularity often found in "traditional" scientific experimental design. In the Lightning workflow, the user is expected to define a "data_setup_function" that is fed into the LightningTrainer(). The user can easily swap out, or modify generation of, the datasets from run-to-run. 
* Automatic GPU support: Lightning allows for easy migration to running training on the GPU. It even allows for multi-GPU training with a simple keyword argument change. Most of all, .to(device) calls are no longer required. 
* Tensorboard and Profiling: The ability to visualize model training using tensorboard, as well as view various PyTorch profiling reports, can also be achieved by simple keyword argument changes. 



#### Data Setup Function

The user is expected to define a *data_setup_function(kwargs)*. This function takes in arbitrary number of keyword arguments (such as number of simualations, batch_size, or other parameters defining the parametric space to sample from) and return for entities: 
1. Train DictDataset -- used for model training. 
2. Dev DictDataset -- used for validation and model checkpointing 
3. Test DictDataset -- currently unsupported
4. Batch Size 

#### Problem Formulation