"""
"""
from copy import deepcopy
import torch
import numpy as np

class Problem:
    def __init__(self, model, optimizer, dataset, logger=None, visualizer=None, epochs=1000):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.logger = logger
        self.visualizer = visualizer
        self.epochs = epochs

    def train(self):
        #######################################
        ### N-STEP AHEAD TRAINING
        #######################################
        best_looploss = np.finfo(np.float32).max
        best_model = deepcopy(self.model.state_dict())

        for i in range(self.epochs):

            self.model.train()
            train_nstep_output = self.model.n_step(self.dataset.train_data)
            self.optimizer.zero_grad()
            train_nstep_output['loss'].backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                dev_nstep_output = self.model.n_step(self.dataset.dev_data)
                dev_loop_output = self.model.loop_step(self.dataset.dev_loop)
                self.logger.log_metrics({**dev_nstep_output, **dev_loop_output}, step=i)
                if dev_loop_output['looploss'] < best_looploss:
                    best_model = deepcopy(self.model.state_dict())
                    best_looploss = dev_loop_output['looploss']
                self.visualizer.plot({**dev_nstep_output, **dev_loop_output}, self.model.state_dict(), best_model)

        plots = self.visualizer.output()
        self.logger.save_artifacts({'best_model.pth': best_model, **plots})

        return best_model





