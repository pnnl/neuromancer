"""
"""
from copy import deepcopy
import torch
import numpy as np
from logger import BasicLogger
from visuals import NoOpVisualizer


class Trainer:
    def __init__(self, model, dataset, optimizer, logger=BasicLogger(), visualizer=NoOpVisualizer(), epochs=1000):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.epochs = epochs

    def train(self):

        best_looploss = np.finfo(np.float32).max
        best_model = deepcopy(self.model.state_dict())
        self.visualizer.train()

        for i in range(self.epochs):

            self.model.train()
            output = self.model.n_step(self.dataset.train_data)
            self.optimizer.zero_grad()
            output['train_nstep_loss'].backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                dev_nstep_output = self.model.n_step(self.dataset.dev_data)
                dev_loop_output = self.model.loop_step(self.dataset.dev_loop)
                self.logger.log_metrics({**dev_nstep_output, **dev_loop_output}, step=i)
                if dev_loop_output['dev_loop_obj_loss'] < best_looploss:
                    best_model = deepcopy(self.model.state_dict())
                    best_looploss = dev_loop_output['dev_loop_obj_loss']
                self.visualizer.plot({**dev_nstep_output, **dev_loop_output}, self.dataset, self.model.state_dict())

        plots = self.visualizer.output()
        self.logger.save_artifacts({'best_model.pth': best_model, **plots})

        return best_model

    def evaluate(self, best_model):

        self.visualizer.eval()
        self.model.load_state_dict(best_model)

        with torch.no_grad():
            ########################################
            ########## NSTEP TRAIN RESPONSE ########
            ########################################
            all_output = dict()
            for dset, dname in zip([self.dataset.train_data, self.dataset.dev_data, self.dataset.test_data],
                                   ['train', 'dev', 'test']):
                output = self.model.n_step(dset)
                self.logger.log_metrics(output)
                all_output = {**all_output, **output}
            self.visualizer.plot(all_output, self.dataset, best_model)

            ########################################
            ########## OPEN LOOP RESPONSE ##########
            ########################################
            all_output = dict()
            for data, dname in zip([self.dataset.train_loop, self.dataset.dev_loop, self.dataset.test_loop],
                                   ['train', 'dev', 'test']):
                output = self.model.loop_step(data)
                self.logger.log_metrics(output)
                all_output = {**all_output, **output}
            self.visualizer.plot(all_output, self.dataset, best_model)
        plots = self.visualizer.output()
        self.logger.log_artifacts(plots)





