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
                 logger: BasicLogger = BasicLogger(), visualizer=Visualizer(), simulator=None, epochs=1000,
                 eval_metric='loop_dev_loss'):
        self.model = problem
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.simulator = simulator
        self.epochs = epochs
        self.logger.log_weights(self.model)
        self.eval_metric = eval_metric

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
                self.logger.log_metrics({**dev_data_output, **dev_sim_output, **output}, step=i)
                if dev_sim_output[self.eval_metric] < best_devloss:
                    best_model = deepcopy(self.model.state_dict())
                    best_devloss = dev_sim_output[self.eval_metric]
                self.visualizer.train_plot({**dev_data_output, **dev_sim_output}, i)

        #   TODO: plot loss function via visualizer
        plots = self.visualizer.train_output()
        self.logger.log_artifacts({'best_model_stat_dict.pth': best_model, **plots})

        # TODO: _pickle.PicklingError: Can't pickle <function <lambda>
        # self.logger.log_artifacts({'best_model_stat_dict.pth': best_model, 'best_model.pth': self.model, **plots})
        # https: // stackoverflow.com / questions / 8804830 / python - multiprocessing - picklingerror - cant - pickle - type - function
        return best_model

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
            ########################################
            ########## SIMULATOR RESPONSE ##########
            ########################################
            test_sim_output = self.simulator.test_eval()
            all_output = {**all_output, **test_sim_output}

        self.all_output = all_output
        self.logger.log_metrics({f'best_{k}': v for k, v in all_output.items()})
        plots = self.visualizer.eval(all_output)
        self.logger.log_artifacts(plots)





