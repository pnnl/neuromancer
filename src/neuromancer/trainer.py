"""


"""
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback


def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


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
        dev_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        logger: BasicLogger = None,
        callback=Callback(),
        lr_scheduler=False,
        epochs=1000,
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
        :param patience: (int) Number of epochs to allow no improvement before early stopping
        :param warmup: (int) How many epochs to wait before enacting early stopping policy
        :param eval_metric: (str) Performance metric for model selection and early stopping
        """
        self.model = problem
        self.optimizer = optimizer
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.callback = callback
        self.logger = logger
        self.epochs = epochs
        self.current_epoch = 0
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

                self.current_epoch = i
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
                    losses = []
                    for d_batch in self.dev_data:
                        d_batch = move_batch_to_device(d_batch, self.device)
                        eval_output = self.model(d_batch)
                        losses.append(eval_output[self.dev_metric])
                    eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
                    output = {**output, **eval_output}
                    self.callback.begin_eval(self, output)  # potential simulator

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
                        print(f'epoch: {i}  {self.train_metric}: {mean_loss}')

                    self.callback.end_eval(self, output)  # visualizations

                    self.callback.end_epoch(self, output)

                    if self.badcount > self.patience:
                        break
        except KeyboardInterrupt:
            print("Interrupted training loop.")

        self.callback.end_train(self, output)  # write training visualizations

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
            self.callback.begin_test(self)  # setup simulator
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

        self.callback.end_test(self, output)    # simulator/visualizations/output concat

        if self.logger is not None:
            self.logger.log_metrics({f"best_{k}": v for k, v in output.items()})

        return output

    def evaluate(self, best_model):
        """
        This method is deprecated. Use self.test instead.
        """
        return self.test(best_model)


def freeze_weight(problem, module_names=['']):
    """
    ['parent->child->child']
    :param component:
    :param module_names:
    :return:
    """
    modules = dict(problem.named_modules())
    for name in module_names:
        freeze_path = name.split('->')
        if len(freeze_path) == 1:
            modules[name].requires_grad_(False)
        else:
            parent = modules[freeze_path[0]]
            freeze_weight(parent, ['->'.join(freeze_path[1:])])


def unfreeze_weight(problem, module_names=['']):
    """
    ['parent->child->child']
    :param component:
    :param module_names:
    :return:
    """
    modules = dict(problem.named_modules())
    for name in module_names:
        freeze_path = name.split('->')
        if len(freeze_path) == 1:
            modules[name].requires_grad_(True)
        else:
            parent = modules[freeze_path[0]]
            freeze_weight(parent, ['->'.join(freeze_path[1:])])
