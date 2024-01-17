## 

## Motivation for PyTorch Lightning 

PyTorch Lightning is a framework built on top of PyTorch designed to simplify the implementation of complex models in PyTorch. It promotes cleaner, more organized code by providing a high-level interface for PyTorch that handles many training intricacies automatically. 

Why might I want to use Lightning in the context of NeuroMANCER? For the user, Lightning simplifies the workflow, allowing one to focus on solving NeuroMANCER problems with their associated datasets. How does it do this? Let's take a look at the added functionality when using Lightning 

## New Features

* PyTorch boilerplate code is removed, thus increasing the accessibility of the NeuroMANCER library. For example, the user doesn't need to deal with details instantiating a PyTorch Dataloader() class. 
* Increased modularity on the data portion: Lightning caters towards modularity often found in "traditional" scientific experimental design. In the Lightning workflow, the user is expected to define a "data_setup_function" that is fed into the LightningTrainer(). The user can easily swap out, or modify generation of, the datasets from run-to-run. 
* Automatic GPU support: Lightning allows for easy migration to running training on the GPU. It even allows for multi-GPU training with a simple keyword argument change. Most of all, .to(device) calls are no longer required. 
* Other features include automatic logging, easy profiling integration, and ability for the user to define own training logic without boilerplate




## Data Setup Function

The user is expected to define a *data_setup_function(kwargs)*. This function takes in arbitrary number of keyword arguments (such as number of simualations, batch_size, or other parameters defining the parametric space to sample from) and return for entities: 
1. Train DictDataset -- used for model training. 
2. Dev DictDataset -- used for validation and model checkpointing 
3. Test DictDataset -- currently unsupported
4. Batch Size 

The signature of this function will look like: 

```
def data_setup_function(*kwargs): 
    # insert data generation code here
```

#### Problem Formulation

For almost all use cases, there is no change in how NeuroMANCER problems are defined when using Lightning. There may be a minor change in how hardcoded tensors need to be handled in the event the user wants to utilize GPU training. For more information on this please refer to the example found in the notebook "Part_2_lightning_advanced_and_gpu_tutorial.ipynb". 

## Training 

The main change when using Lightning is how Problem training gets done. Currently there is a Neuromancer Trainer() classes that handles training a Problem on various dataloaders. As an alternative training mechanism, we introduce a Lightning Trainer (LitTrainer) class which is designed to simplify the training process for the user as well as include a couple more key features. 

The user will invoke simply invoke:

```
lit_trainer = LitTrainer(*kwargs)
lit_trainer.fit(problem, data_setup_function, *kwargs)
```

Which will then train that particular Problem to data governed by the data_setup_function. 


## Tutorials 

The user will find several Lightning-Neuromancer tutorials in this folder. We describe the purpose of each here:
* Part 1: Goes over how basics on how a Neuromancer set-up can be converted into the Lightning version
* Part 2: Goes over more advanced / nuanced cases when migrating towards Lightning. Also showcases automated GPU support 
* Part 3: Goes over solving a PINN with the Lightning workflow 
* Part 4: Goes over using Koopman Operators with the Lightning workflow. Also showcases how to easily visualize training progress with Tensorboard
* Part 5: Goes over using cvxpy with Lightning workflow. 
* Part 6: Is a Python script that demonstrates solving a computationally expensive problem with automated multi-GPU distributed training 
* Part 7: Demonstrates how to easily profile/benchmark  


## LitTrainer Parameters
* epochs: Number of training epochs 
* monitor_metric: One of 'train_loss', 'dev_loss', 'test_loss'. This metric will be monitored every epoch during training and will save the best weights accordingly. 
* dev_metric: 
* custom_optimizer: 
    * If the user wants to pass in their own optimizer. For example: 
    ```
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    lit_trainer = LitTrainer(custom_optimizer=optimizer)
    ```
    By default the optimizer used is: `torch.optim.Adam(self.problem.parameters(), 0.001, betas=(0.0, 0.9))`

* devices: Please refer to "Device Management" below
* strategy: Please refer to "Device Management" below 
* accelerator: Please refer to "Device Management" below. 
* profiler: Lightning integrates easily with PyTorch profilers such as "simple" or "pytorch"

## Example
The following is basic pseudocode that outlines the steps required to utilize this Lightning integration. In this example, we define a data_setup_function (DSF) that takes in "nsim" as an argument and returns the Neuromancer DictDatasets (which take in nsim as a parameter) and a batch_size. We then instantiate a LitTrainer to run for 10 epochs on the GPU with device=1 (akin to cuda:1) and train it on a problem and DSF
```
def data_setup_function(nsim=5000): 
    train_data = DictDataset(nsim, ... , name='train')
    dev_data = DictDataset(nsim, ... , name='dev)
    batch_size = 32
    return train_data, dev_data, None, batch_size 

problem = Problem(...)
lit_trainer = LitTrainer(epochs=10, accelerator='gpu', devices=[1])
lit_trainer.fit(problem, data_setup_function, nsim=100)
```





## Device Management
#### To Run on CPU: 
* accelerator = "cpu" is all that is necessary

#### To Run on GPU: 
* accelerator = "gpu" is required 
* devices can be a list, integer, or "auto"
    * ex) [1,2,3] will distribute training over cuda:1, cuda:2, and cuda:3
    * ex) 7 will distribute training over the 7 GPUs automatically selected 
    * ex) "Auto" for automatic selection based on the chosen accelerator. We do not recommend this. 
* strategy is either "auto", "ddp" or "ddp_notebook"
    * "auto" will utilize a single GPU 
    * "ddp" will run distributed training across devices desginated under "devices" assuming len(devices) > 1. This keyword should *NOT* be used in notebooks, only scripts 
    * "ddp_notebook" is akin to "ddp" and should only be used in notebook environments. Note this feature is currently unstable according to Lightning developers and should be avoided. For safety, please only use "ddp" strategy
    
    

For more information please see: https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api


## Saving and Loading Problem Weights
foo

## Other Features 

### Tensorboard

### Profiling 

### Custom Training Logic



