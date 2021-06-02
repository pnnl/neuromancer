from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from neuromancer.loggers import BasicLogger
from neuromancer.visuals import Visualizer
from neuromancer.problem import Problem
from neuromancer.datasets import Dataset
from neuromancer.simulators import Simulator
from neuromancer.callbacks import SysIDCallback


class TrainerMPP:
    def __init__(
        self,
        problem: Problem,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        logger: BasicLogger = None,
        visualizer=Visualizer(),
        epochs=1000,
        eval_metric="dev_loss",
    ):
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
            output["train_loss"].backward()
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
        self.logger.log_artifacts({"best_model_stat_dict.pth": best_model, **plots})
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
            splits = [
                self.dataset.train_data,
                self.dataset.dev_data,
                self.dataset.test_data,
            ]
            for dset, dname in zip(splits, ["train", "dev", "test"]):
                all_output = {**all_output, **self.model(dset)}

        self.all_output = all_output
        self.logger.log_metrics({f"best_{k}": v for k, v in all_output.items()})
        if self.visualizer is not None:
            plots = self.visualizer.eval(all_output)
            self.logger.log_artifacts(plots)



