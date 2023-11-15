"""


"""
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

import numpy as np

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class LitProblem(pl.LightningModule): 
    def __init__(self, problem, train_metric='train_loss', dev_metric='dev_loss', test_metric='test_loss'): 
        super().__init__()
        self.problem = problem 
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        
        self.training_step_outputs = []
    
    def training_step(self, batch): 
        output = self.problem(batch)
        loss = output[self.train_metric]
        self.training_step_outputs.append(loss)
        self.log('train_loss', loss, on_epoch=True, enable_graph=True, prog_bar=False)
        return loss 
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        print("EPOCH AVERAGE ", epoch_average)
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory

    """
    def validation_step(self, batch): 
        output = self.problem(batch)
        assert self.dev_metric in output, f"Error: {self.dev_metric} not found in problem output"
        loss = output[self.dev_metric]
        self.log('dev_loss', loss)
    """
    def configure_optimizers(self): 
      
        optimizer = torch.optim.Adam(self.problem.parameters(), 0.001, betas=(0.0, 0.9))
        return optimizer 



class LightningTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lit_problem = None 

        self.epochs = 1000
        self.train_metric = 'train_loss'
        self.dev_metric = 'dev_loss'
        self.test_metric = 'test_loss'
        self.eval_metric = 'dev_loss'
        self.patience = 5 
        self.warmup = 0 
        self.clip = 100.0

        self.setup_attributes(*args, **kwargs)

        self.model_checkpoint = ModelCheckpoint(
            save_weights_only=True,
            monitor=self.dev_metric,
            mode='min',
            save_top_k=1,
            verbose=True
        )

    def setup_attributes(self, *args, **kwargs): 
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
     
        if 'train_metric' in kwargs:
            self.train_metric = kwargs['train_metric']

        if 'dev_metric' in kwargs:
            self.dev_metric = kwargs['dev_metric']

        if 'test_metric' in kwargs:
            self.test_metric = kwargs['test_metric']

        if 'eval_metric' in kwargs:
            self.eval_metric = kwargs['eval_metric']

        if 'patience' in kwargs:
            self.patience = kwargs['patience']

        if 'warmup' in kwargs:
            self.warmup = kwargs['warmup']

        if 'clip' in kwargs:
            self.clip = kwargs['clip']


    def get_weights(self):
        # Implement your custom method logic here
        best_model = self.model_checkpoint.best_model_path
        return best_model.state_dict()

    def fit(self, problem, datamodule):

        if self.lit_problem is None: 
            self.lit_problem = LitProblem(problem,self.train_metric, self.dev_metric, self.test_metric )
        # Override the fit method if needed
        super().fit(problem, datamodule)
        # Add custom logic here if necessary

    
    


class Trainer:
    """
    Class encapsulating boilerplate PyTorch training code. Training procedure is somewhat
    extensible through methods in Callback objects associated with training and evaluation
    waypoints.
    """
    def __init__(
        self,
        problem: Problem,
        train_data: torch.utils.data.DataLoader,
        dev_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
        optimizer: torch.optim.Optimizer = None,
        logger: BasicLogger = None,
        callback=Callback(),
        lr_scheduler=False,
        epochs=1000,
        epoch_verbose=1,
        patience=5,
        warmup=0,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        eval_mode="min",
        clip=100.0,
        device="cpu"
    ):
        """

        :param problem: Object which defines multi-objective loss function and computational graph
        :param dataset: Batched (over chunks of time if sequence data) dataset for non-stochastic gradient descent
        :param optimizer: Pytorch optimizer
        :param logger: Object for logging results
        :param epochs: (int) Number of epochs to train
        :param epoch_verbose (int) printing epoch metric at each i-th epoch
        :param patience: (int) Number of epochs to allow no improvement before early stopping
        :param warmup: (int) How many epochs to wait before enacting early stopping policy
        :param eval_metric: (str) Performance metric for model selection and early stopping
        """
        self.model = problem
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(problem.parameters(), 0.01, betas=(0.0, 0.9))
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.callback = callback
        self.logger = logger
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        if logger is not None:
            self.logger.log_weights(self.model)
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        self.eval_metric = eval_metric
        self._eval_min = eval_mode == "min"
        self.lr_scheduler = (
            ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=100)
            if lr_scheduler
            else None
        )
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0
        self.clip = clip
        self.best_devloss = np.finfo(np.float32).max if self._eval_min else 0.
        self.best_model = deepcopy(self.model.state_dict())
        self.device = device

    def train(self):
        """
        Optimize model according to train_metric and validate per-epoch according to eval_metric.
        Trains for self.epochs and terminates early if self.patience threshold is exceeded.
        """
        self.callback.begin_train(self)

        try:
            for i in range(self.current_epoch, self.current_epoch+self.epochs):

                self.model.train()
                losses = []
                for t_batch in self.train_data:
                    t_batch['epoch'] = i
                    t_batch = move_batch_to_device(t_batch, self.device)
                    output = self.model(t_batch)
                    self.optimizer.zero_grad()
                    output[self.train_metric].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    losses.append(output[self.train_metric])
                    self.callback.end_batch(self, output)

                output[f'mean_{self.train_metric}'] = torch.mean(torch.stack(losses))
                self.callback.begin_epoch(self, output)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(output[f'mean_{self.train_metric}'])

                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data is not None:
                        losses = []
                        for d_batch in self.dev_data:
                            d_batch = move_batch_to_device(d_batch, self.device)
                            eval_output = self.model(d_batch)
                            losses.append(eval_output[self.dev_metric])
                        eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
                        output = {**output, **eval_output}
                    self.callback.begin_eval(self, output)  # Used for alternate dev evaluation

                    if (self._eval_min and output[self.eval_metric] < self.best_devloss)\
                            or (not self._eval_min and output[self.eval_metric] > self.best_devloss):
                        self.best_model = deepcopy(self.model.state_dict())
                        self.best_devloss = output[self.eval_metric]
                        self.badcount = 0
                    else:
                        if i > self.warmup:
                            self.badcount += 1
                    if self.logger is not None:
                        self.logger.log_metrics(output, step=i)
                    else:
                        mean_loss = output[f'mean_{self.train_metric}']
                        if i % (self.epoch_verbose) == 0:
                            print(f'epoch: {i}  {self.train_metric}: {mean_loss}')

                    self.callback.end_eval(self, output)  # visualizations

                    self.callback.end_epoch(self, output)

                    if self.badcount > self.patience:
                        print('Early stopping!!!')
                        break
                    self.current_epoch = i + 1

        except KeyboardInterrupt:
            print("Interrupted training loop.")

        self.callback.end_train(self, output)  # write training visualizations

        # Assign best weights to the model
        self.model.load_state_dict(self.best_model)

        if self.logger is not None:
            self.logger.log_artifacts({
                "best_model_state_dict.pth": self.best_model,
                "best_model.pth": self.model,
            })
        return self.best_model

    def test(self, best_model):
        """
        Evaluate the model on all data splits.
        """
        self.model.load_state_dict(best_model, strict=False)
        self.model.eval()

        with torch.set_grad_enabled(self.model.grad_inference):
            self.callback.begin_test(self)
            output = {}
            for dset, metric in zip([self.train_data, self.dev_data, self.test_data],
                                    [self.train_metric, self.dev_metric, self.test_metric]):
                losses = []
                for batch in dset:
                    batch = move_batch_to_device(batch, self.device)
                    batch_output = self.model(batch)
                    losses.append(batch_output[metric])
                output[f'mean_{metric}'] = torch.mean(torch.stack(losses))
                output = {**output, **batch_output}

        self.callback.end_test(self, output)

        if self.logger is not None:
            self.logger.log_metrics({f"best_{k}": v for k, v in output.items()})

        return output

    def evaluate(self, best_model):
        """
        This method is deprecated. Use self.test instead.
        """
        return self.test(best_model)