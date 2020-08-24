"""

"""
# python base imports
from copy import deepcopy

# machine learning/data science imports
import torch
import numpy as np

# local imports
from neuromancer.loggers import BasicLogger
from neuromancer.visuals import Visualizer
from neuromancer.problem import Problem
from neuromancer.datasets import Dataset
from neuromancer.simulators import Simulator


def reset(module):
    for mod in module.modules():
        if hasattr(mod, 'reset') and mod is not module:
            mod.reset()


class Trainer:

    def __init__(self, problem: Problem,
                 dataset: Dataset,
                 optimizer: torch.optim.Optimizer,
                 logger: BasicLogger = None,
                 visualizer: Visualizer = None,
                 simulator: Simulator = None,
                 epochs=1000, eval_metric='loop_dev_loss', patience=5,
                 warmup=0):
        """

        :param problem: Object which defines multi-objective loss function and computational graph
        :param dataset: Batched (over chunks of time if sequence data) dataset for non-stochastic gradient descent
        :param optimizer: Pytorch optimizer
        :param logger: Object for logging results
        :param visualizer: Object for creating visualizations of the model during and after training
        :param simulator: Object for simulating learned model for assessing performance
        :param epochs: (int) Number of epochs to train
        :param eval_metric: (str) Performance metric for model selection and early stopping
        :param patience: (int) Number of epochs to allow no improvement before early stopping
        :param warmstart: (int) How many epochs to wait before enacting early stopping policy
        """
        self.model = problem
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.simulator = simulator
        self.epochs = epochs
        self.logger.log_weights(self.model)
        self.eval_metric = eval_metric
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0

    ########################################
    ############# TRAIN LOOP ###############
    ########################################
    def train(self):
        best_devloss = np.finfo(np.float32).max
        best_model = deepcopy(self.model.state_dict())
        for i in range(self.epochs):
            self.model.train()
            output = self.model(self.dataset.train_data)
            self.optimizer.zero_grad()
            output['nstep_train_loss'].backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                dev_data_output = self.model(self.dataset.dev_data)
                dev_sim_output = self.simulator.dev_eval()
                output = {**output, **dev_data_output, **dev_sim_output}
                if output[self.eval_metric] < best_devloss:
                    best_model = deepcopy(self.model.state_dict())
                    best_devloss = output[self.eval_metric]
                    self.badcount = 0
                else:
                    if i > self.warmup:
                        self.badcount += 1
                self.logger.log_metrics(output, step=i)
                self.visualizer.train_plot(output, i)
            if self.badcount > self.patience:
                break

        plots = self.visualizer.train_output()
        self.logger.log_artifacts({'best_model_state_dict.pth': best_model, 'best_model.pth': self.model, **plots})
        return best_model

    ########################################
    ########## EVALUATE MODEL ##############
    ########################################
    def evaluate(self, best_model):
        self.model.eval()
        self.model.load_state_dict(best_model)
        self.simulator.model.load_state_dict(best_model)

        with torch.no_grad():
            ########################################
            ########### DATASET RESPONSE ###########
            ########################################
            all_output = dict()
            for dset, dname in zip([self.dataset.train_data, self.dataset.dev_data, self.dataset.test_data],
                                   ['train', 'dev', 'test']):
                all_output = {**all_output, **self.model(dset)}
            ########################################
            ########## SIMULATOR RESPONSE ##########
            ########################################
            test_sim_output = self.simulator.test_eval()
            all_output = {**all_output, **test_sim_output}

        self.all_output = all_output
        self.logger.log_metrics({f'best_{k}': v for k, v in all_output.items()})
        plots = self.visualizer.eval(all_output)
        self.logger.log_artifacts(plots)
        return all_output


class TrainerMPP:
    def __init__(self, problem: Problem, dataset: Dataset, optimizer: torch.optim.Optimizer,
                 logger: BasicLogger = None, visualizer=Visualizer(), epochs=1000,
                 eval_metric='dev_loss'):
        self.model = problem
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.epochs = epochs
        self.logger.log_weights(self.model)
        self.eval_metric = eval_metric

    ########################################
    ############# TRAIN LOOP ###############
    ########################################
    def train(self):
        best_devloss = np.finfo(np.float32).max
        best_model = deepcopy(self.model.state_dict())
        best_model_full = self.model
        for i in range(self.epochs):
            self.model.train()
            output = self.model(self.dataset.train_data)
            self.optimizer.zero_grad()
            output['train_loss'].backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                dev_data_output = self.model(self.dataset.dev_data)
                self.logger.log_metrics({**dev_data_output, **output}, step=i)
                if dev_data_output[self.eval_metric] < best_devloss:
                    best_model = deepcopy(self.model.state_dict())
                    best_model_full = self.model
                    best_devloss = dev_data_output[self.eval_metric]
                self.visualizer.train_plot(dev_data_output, i)

        plots = self.visualizer.train_output()
        self.logger.log_artifacts({'best_model_stat_dict.pth': best_model, **plots})
        return best_model, best_model_full

    ########################################
    ########## EVALUATE MODEL ##############
    ########################################
    def evaluate(self, best_model):
        self.model.eval()
        self.model.load_state_dict(best_model)

        with torch.no_grad():
            ########################################
            ########### DATASET RESPONSE ###########
            ########################################
            all_output = dict()
            for dset, dname in zip([self.dataset.train_data, self.dataset.dev_data, self.dataset.test_data],
                                   ['train', 'dev', 'test']):
                all_output = {**all_output, **self.model(dset)}

        self.all_output = all_output
        self.logger.log_metrics({f'best_{k}': v for k, v in all_output.items()})
        plots = self.visualizer.eval(all_output)
        self.logger.log_artifacts(plots)





