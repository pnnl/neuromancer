"""
"""
from copy import deepcopy
import torch
import numpy as np
from logger import BasicLogger
from visuals import Visualizer
from problem import Problem
from datasets import Dataset


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

            # TODO: HACK
            # TODO: create eval dataset method linked with the dataset
            with torch.no_grad():
                self.model.eval()
                dev_nstep_output = self.model(self.dataset.dev_data)
                if self.dataset.type == 'openloop':
                    dev_loop_output = self.model(self.dataset.dev_loop)
                    self.logger.log_metrics({**dev_nstep_output, **dev_loop_output, **output}, step=i)
                elif self.dataset.type == 'closedloop':
                # # TODO: placeholder
                    dev_loop_output = {'loop_dev_loss': 0}
                    self.logger.log_metrics({**dev_nstep_output,  **output}, step=i)
                if dev_loop_output['loop_dev_loss'] < best_looploss:
                    best_model = deepcopy(self.model.state_dict())
                    best_looploss = dev_loop_output['loop_dev_loss']
                self.visualizer.train_plot({**dev_nstep_output, **dev_loop_output}, i)

        #   TODO: plot loss function via visualizer
        plots = self.visualizer.train_output()
        self.logger.log_artifacts({'best_model_stat_dict.pth': best_model, **plots})

        # TODO: _pickle.PicklingError: Can't pickle <function <lambda>
        # self.logger.log_artifacts({'best_model_stat_dict.pth': best_model, 'best_model.pth': self.model, **plots})
        # https: // stackoverflow.com / questions / 8804830 / python - multiprocessing - picklingerror - cant - pickle - type - function
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

            # TODO: keep only nstep train response in trainer?
            # TODO: should we create standalone simulator class for OL and CL responses?
            ########################################
            ########## OPEN LOOP RESPONSE ##########
            ########################################
            if self.dataset.type == 'openloop':
                for data, dname in zip([self.dataset.train_loop, self.dataset.dev_loop, self.dataset.test_loop],
                                       ['train', 'dev', 'test']):
                    all_output = {**all_output, **self.model(data)}

            ########################################
            ########## CLOSED LOOP RESPONSE ########
            ########################################
            elif self.dataset.type == 'closedloop':
                pass
                #  TODO: simulate closed loop with emulators or trained model

        self.all_output = all_output
        self.logger.log_metrics({f'best_{k}': v for k, v in all_output.items()})
        plots = self.visualizer.eval(all_output)
        self.logger.log_artifacts(plots)





