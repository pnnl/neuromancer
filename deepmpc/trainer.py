"""
"""
from copy import deepcopy
import torch
import numpy as np
from logger import BasicLogger
from visuals import Visualizer
from loops import Problem
from dataset import Dataset


def reset(module):
    for mod in module.modules():
        if hasattr(mod, 'reset') and mod is not module:
            mod.reset()


class Trainer:

    def __init__(self, problem: Problem, dataset: Dataset, optimizer: torch.optim.Optimizer,
                 logger: BasicLogger = BasicLogger(), visualizer=Visualizer(), epochs=1000):
        self.model = problem
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.epochs = epochs
        self.logger.log_weights(self.model)

    def train(self):

        best_looploss = np.finfo(np.float32).max
        best_model = deepcopy(self.model.state_dict())

        for i in range(self.epochs):
            self.model.train()
            output = self.model(self.dataset.train_data)
            self.optimizer.zero_grad()
            output['nstep_train_loss'].backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                dev_nstep_output = self.model(self.dataset.dev_data)
                dev_loop_output = self.model(self.dataset.dev_loop)
                self.logger.log_metrics({**dev_nstep_output, **dev_loop_output, **output}, step=i)
                if dev_loop_output['loop_dev_loss'] < best_looploss:
                    best_model = deepcopy(self.model.state_dict())
                    best_looploss = dev_loop_output['loop_dev_loss']
                self.visualizer.train_plot({**dev_nstep_output, **dev_loop_output}, i)
        #   TODO: add inputs to visualizer
        #   TODO: add loss function to visualizer

        plots = self.visualizer.train_output()
        self.logger.log_artifacts({'best_model.pth': best_model, **plots})

        return best_model

    def evaluate(self, best_model):
        self.model.eval()
        self.model.load_state_dict(best_model)

        with torch.no_grad():
            ########################################
            ########## NSTEP TRAIN RESPONSE ########
            ########################################
            all_output = dict()
            for dset, dname in zip([self.dataset.train_data, self.dataset.dev_data, self.dataset.test_data],
                                   ['train', 'dev', 'test']):
                all_output = {**all_output, **self.model(dset)}

            # TODO: should we create standalone simulator class for OL and CL?
            # TODO: keep only train response in trainer?

            ########################################
            ########## OPEN LOOP RESPONSE ##########
            ########################################
            for data, dname in zip([self.dataset.train_loop, self.dataset.dev_loop, self.dataset.test_loop],
                                   ['train', 'dev', 'test']):
                all_output = {**all_output, **self.model(data)}

            ########################################
            ########## CLOSED LOOP RESPONSE ########
            ########################################
        #     TODO: simulate closed loop with emulators or trained model

        self.all_output = all_output
        self.logger.log_metrics({f'best_{k}': v for k, v in all_output.items()})
        #  TODO: add inputs to visualizer
        plots = self.visualizer.eval(all_output)
        self.logger.log_artifacts(plots)





