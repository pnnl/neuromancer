"""
The following is a Neuromancer example that demonstrates the need for distributed training across GPUs.
The objective function here is complex and this example cannot be solved on the CPU without error. 
Note that this script still takes about ~90 minutes to run on 2 GPUs. Most of that time is "setup time"/preprocessing.
The training loop should execute in a manner of seconds once the preprocessing is complete
"""



import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import neuromancer as nm
from neuromancer.dataset import DictDataset

import lightning.pytorch as pl 
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

from neuromancer.trainer import Trainer, LitTrainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node


# Define the data setup function
def data_setup_function(exp_returns): 
        p_low, p_high = max(min(exp_returns),0), max(exp_returns)
        data_train = DictDataset({"p": torch.FloatTensor(1000, 1).uniform_(p_low, p_high)})
        data_test = DictDataset({"p": torch.FloatTensor(100, 1).uniform_(p_low, p_high)})
        data_dev = DictDataset({"p": torch.FloatTensor(100, 1).uniform_(p_low, p_high)})
        return data_train, data_dev, data_test, 32

def main(): 
        num_vars = 5

        # expected returns
        exp_returns = np.random.uniform(0.002, 0.01, num_vars)
        print("Expected Returns:")
        print(exp_returns)

        # covariance matrix
        A = np.random.rand(num_vars, num_vars)
        # positive semi-definite matrix
        cov_matrix = A @ A.T / 1000
        print("Covariance Matrix:")
        print(cov_matrix)


        # parameters
        p = nm.constraint.variable("p")
        # variables
        x = nm.constraint.variable("x")

        # objective function
        f = sum(cov_matrix[i, j] * x[:, i] * x[:, j] for i in range(num_vars) for j in range(num_vars))
        obj = [f.minimize(weight=1.0, name="obj")]

        # constraints
        constrs = []
        # constr: 100 units
        con = 100 * (sum(x[:, i] for i in range(num_vars)) == 1)
        con.name = "c_units"
        constrs.append(con)
        # constr: expected return
        con = 100 * (sum(exp_returns[i] * x[:, i] for i in range(num_vars)) >= p[:, 0])
        con.name = "c_return"
        constrs.append(con)

        # define neural architecture for the solution map
        func = nm.modules.blocks.MLP(insize=1, outsize=num_vars, bias=True,
                                linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[5]*4)
        # solution map from model parameters: sol_map(p) -> x
        sol_map = nm.system.Node(func, ["p"], ["x"], name="smap")
        # trainable components
        components = [sol_map]

        # merit loss function
        loss = nm.loss.PenaltyLoss(obj, constrs)
        # problem
        problem = nm.problem.Problem(components, loss)

        # training
        lr = 0.001  # step size for gradient descent

        # set adamW as optimizer
        optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)

        # Define lightning trainer. We use GPU acceleration utilizing 2 GPUS. We tell Lightning to 
        # distribute training parallely (strategy=ddp)
        lit_trainer = LitTrainer(epochs=10, accelerator="gpu", devices=[1,2], strategy="ddp", dev_metric='train_loss')

        # Train problem to the data_setup_function
        lit_trainer.fit(problem, data_setup_function, exp_returns=exp_returns)

if __name__ == "__main__": 
        main()
        