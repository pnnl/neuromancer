# Additional Instructions for Lightning Integration Features

## Motivation for PyTorch Lightning 

PyTorch Lightning is a framework built on top of PyTorch designed to simplify the implementation of complex models in PyTorch. It promotes cleaner, more organized code by providing a high-level interface for PyTorch that handles many training intricacies automatically. 

Why might I want to use Lightning in the context of NeuroMANCER? For the user, Lightning simplifies the workflow, allowing one to focus on solving NeuroMANCER problems with their associated datasets. How does it do this? Let's take a look at the added functionality when using Lightning. 

## New Features

* PyTorch boilerplate code is removed, thus increasing the accessibility of the NeuroMANCER library. For example, the user doesn't need to deal with details instantiating a PyTorch Dataloader() class. 
* Increased modularity on the data portion: Lightning caters towards modularity often found in "traditional" scientific experimental design. In the Lightning workflow, the user is expected to define a "data_setup_function" that is fed into the LightningTrainer(). The user can easily swap out, or modify generation of, the datasets from run-to-run. 
* Automatic GPU support: Lightning allows for easy migration to running training on the GPU. It even allows for multi-GPU training with a simple keyword argument change. Most of all, .to(device) calls are no longer required. 
* Other features include hyperparameter tuning with wandb, automatic logging, easy profiling integration, and ability for the user to define own training logic without boilerplate




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

## Problem Formulation

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

The user will find several Lightning-Neuromancer tutorials in this folder. There are two main tutorials
* Part 1: Goes over how basics on how a Neuromancer set-up can be converted into the Lightning version
* Part 2: Goes over more advanced / nuanced cases when migrating towards Lightning. Also showcases automated GPU support, loading/saving weights
* Part 3: Is a Python script that demonstrates solving a computationally expensive problem with automated multi-GPU distributed training

Other domain-specific examples can be found in the "other_examples" folder: 

* lightning_nonauto_DeepKoopman: Goes over using Koopman Operators with the Lightning workflow. Also showcases how to easily visualize training progress with Tensorboard
* lightning_cvxpy_layers: Goes over using cvxpy with Lightning workflow. 
* lighting_custom_training_example: Demonstrates how the user can define their own training logic to replace default training logic, if desired




## LitTrainer Parameters
* epochs: Number of training epochs 
* eval_metric: This metric will be monitored every epoch during training and will save the best weights accordingly. Default 'dev_loss'
* train_metric: Metric for training. Defaults to 'train_loss'.
* dev_metric: Metric for development/validation. Defaults to 'dev_loss'.
* test_metric: Metric for testing. Defaults to 'test_loss'. Currently unused
* patience: Number of epochs to wait for improvement before early stopping. Defaults to None (no patience)
* warmup: Number of warmup epochs. Defaults to 0.
* clip: Gradient clipping value, by norm. Defaults to 100.0
* custom_optimizer: 
    * If the user wants to pass in their own optimizer. For example: 
    ```
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    lit_trainer = LitTrainer(custom_optimizer=optimizer)
    ```
    By default the optimizer used is: `torch.optim.Adam(self.problem.parameters(), 0.001, betas=(0.0, 0.9))`

* save_weights: Set to True if best Problem weights should be saved to disk. Default true 
* weight_path: Folder to save weights, defaults to ./
* weight_name: Name of the weight file. By default, filename is None and will be set to '{epoch}-{step}', where “epoch” and “step” match the number of finished epoch and optimizer steps respectively.
* devices: Please refer to "Device Management" below
* strategy: Please refer to "Device Management" below 
* accelerator: Please refer to "Device Management" below. 
* profiler: Lightning integrates easily with PyTorch profilers such as "simple" or "pytorch"
* custom_training_step: Custom training step function, if desired. Defaults to None, in which case the standard training step procedure is executed. See Custom Training Logic section below

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
    * "auto" will utilize whatever hardward is "best" available  
    * "ddp" will run distributed training across devices desginated under "devices" assuming len(devices) > 1. This keyword should *NOT* be used in notebooks, only scripts 
    * "ddp_notebook" is akin to "ddp" and should only be used in notebook environments
    
    
For more information please see: https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api


## Saving and Loading Problem Weights
By default, Problems() passed into LitTrainer (as well as base Neuromancer trainer) will automatically have the best weights at end of training, so there should be no need to manually load best_weights at the end of training.

That said, we can save and load weights as follows: 
* Set save_weights argument to True (this is default)
* Specify directory where to save weights (optional, by default is the current working directory)
* Use *load_state_dict_lightning()* function to properly ingest weights into Problem

By default, weights will be saved with the following convention: '{epoch}-{step}.ckpt', where “epoch” and “step” match the number of finished epoch and optimizer steps respectively. The weights file can be given a custom name by changing the "weight_name" argument to LiTrainer. E.g. "test_weights" will save to "test_weights.ckpt"

For example, the following code would save the weights to ./test_weights.ckpt. It then loads the weights into a desired Problem using load_state_dict_lightning()
```
lit_trainer = LitTrainer(epochs=200, accelerator='cpu', custom_optimizer=optimizer, monitor_metric='dev_loss', weight_name='test_weights')
lit_trainer.fit(problem, data_setup_function)
```

Load weights. Note that unless the problem specified here is untrained, this step is redundant
```
load_state_dict_lightning(problem, 'test_weights.ckpt')
```

# Other Features 

## Wandb Hyperparameter Tuning 


We now will tune learning rate and batch size hyperparameters. Recall in previous tutorials we use the `LitTrainer.fit()` function to fit a problem to data_setup_function. To tune hyperparameters, we use `LitTrainer.hyperparameter_sweep()` instead. The syntax is similar: 

### Fit

```bash
lit_trainer = LitTrainer()
lit_trainer.fit(problem, data_setup_function, *kwargs)
```

### Tuning 

```bash
lit_trainer = LitTrainer()
lit_trainer.hyperparameter_sweep(problem, data_setup_function, sweep_config, *kwargs)
```

The only difference is the addition of a sweep configuration file in the format required by wandb. An example of such a config is shown below: 
```bash
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {
            'min': 0.001,
            'max': .007
        },
        'batch_size': {
            'values': [16, 64, 128]
        }
    }
}
```

We go over the parameters for sweep below. Again the function signature is: 

```bash
def hyperparameter_sweep(self, problem, data_setup_function, sweep_config, count=10, project_name='run_sweep', **kwargs):
```
* problem: Neuromancer problem 
* data_setup_function: A data setup function 
* sweep_config: Dictionary of sweep parameters. 
* Count: Number of iterations to sample from the param distributions. E.g. 10 will execute 10 runs 
* project_name: Name of the project. Generally unnecesssary
* **kwargs: Any keyword arguments needed for data setup function 

### WandB Set-Up (Deprecated For Now, Will be re-added in future release)

Please ensure you have a wandb account setup and provided API key. When running on VS Code, one will need to provide said API key upon launching hyperparameter_sweep()

### WandB Visualization

Upon launching the sweep, we will see a supplied link(s) to view the sweeps and runs in stdout

For more information please refer to: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration



## Tensorboard
Lightning automatically will log training history to a *lightning_logs* found in the current working direcotyr. The latest "version" should correspond to the most current training run. As a result it is easy to view training progress, etc. with Tensorboard. For example, assuming one is in VS Code environment with Tensorboard plug-in installed, one can launch a Tensorboard session to view latest training progress with: 
```
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
``````
## Profiling 
One can profile training run easily by passing in the "profiler" keyword argument to LitTrainer, for example: 

```
lit_trainer = LitTrainer(profiler='simple)
```

Will output profiling report at end of training. 
Profiling options include "simple", "pytorch" and "advanced". For more information please see https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/profiler.html#pytorch-profiling

## Custom Training Logic
Training within PyTorch Lightning framework is defined by a `training_step` function, which defines the logic going from a data batch to loss. For example, the default training_step used is shown below (other extraneous details removed for simplicity). Here, we get the problem output for the given batch and return the loss associated with that output.

```
def training_step(self, batch):
    output = self.problem(batch)
    loss = output[self.train_metric]
    return loss
```
While rare, there may be instances where the user might want to define their own training logic. Potential cases include test-time data augmentation (e.g. operations on/w.r.t the data rollout), other domain augmentations, or modifications to how the output and/or loss is handled. 

The user can pass in their own "training_step" by supplying an equivalent function handler to the "custom_training_step" keyword of LitTrainer, for example: 

```
def custom_training_step(model, batch): 
    output = model.problem(batch)
    Q_con = 1
    if model.current_epoch > 1: 
        Q_con = 1 
    loss = Q_con*(output[model.train_metric])
    return loss
```

The signature of this function should be `custom_training_step(model, batch)` where model is a Neuromancer Problem. 

We are not limited to just implementing custom training step. Lightning provides a plethora of "hooks" for the user to interact with in modular "lego" style fashion that is intrinsic to the Lightning framework. 

Please refer to lightning_integration_examples/other_examples/custom_training_example.ipynb for more information on these flexible training options. 
