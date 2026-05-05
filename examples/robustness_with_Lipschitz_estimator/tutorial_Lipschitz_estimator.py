# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
import torch
import os

from eclipse_nn.LipConstEstimator import LipConstEstimator


## Model1: create estimator by torch model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


# torch model
model_t = SimpleNet()
# create estimator with the model
est_t = LipConstEstimator(model=model_t)
# show model information (optional)
est_t.model_review()


## Model 2: create estimator by given weights
import requests
import numpy as np
import torch

# Raw URL of weights.npz (replace with your actual raw link)
url = "https://raw.githubusercontent.com/YuezhuXu/ECLipsE/498af938ce64b5a0c9da9864a90ba43ef1afcb9c/sampleweights/npz/lyr2n80test1.npz"

# Download and save the file locally
response = requests.get(url)
response.raise_for_status()
with open("weights.npz", "wb") as f:
    f.write(response.content)

# Load the npz
weights_npz = np.load("weights.npz")

weights = []
wkeys = sorted(
    [k for k in weights_npz.files if k.startswith("w")],
    key=lambda k: int(k[1:])  # numeric sort by suffix
)

for k in wkeys:
    Wi = torch.tensor(weights_npz[k], dtype=torch.float64)
    weights.append(Wi)
    
est_w = LipConstEstimator(weights=weights)

## Model 3: create stimator by nothing (randomly generate a FNN)
est_n = LipConstEstimator()
# assign layer information
est_n.generate_random_weights([10,40,3])


## Assign the method for all the three models and estimate
## Model 1
lip_trivial_t = est_t.estimate(method='trivial')
lip_fast_t = est_t.estimate(method='ECLipsE_Fast')
lip_acc_t = est_t.estimate(method='ECLipsE') # dies

# show results
print('For estimator created by a torch model.')
print(f'Trivial Lipschitz estimate = {lip_trivial_t}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_t}')
print(f'Lipschitz estimate from EClipsE = {lip_acc_t}')

# relative tightness
print(f'Ratio Fast = {lip_fast_t / lip_trivial_t}')
print(f'Ratio Acc = {lip_acc_t / lip_trivial_t}')


## Model 2
lip_trivial_w = est_w.estimate(method='trivial')
lip_fast_w = est_w.estimate(method='ECLipsE_Fast')
lip_acc_w = est_w.estimate(method='ECLipsE')

# show results
print("For estimator created by given weights.")
print(f'Trivial Lipschitz estimate = {lip_trivial_w}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_w}')
print(f'Lipschitz estimate from EClipsE = {lip_acc_w}')


# relative tightness
print(f'Ratio Fast = {lip_fast_w / lip_trivial_w}')
print(f'Ratio Acc = {lip_acc_w / lip_trivial_w}')


## Model 3
lip_trivial_n = est_n.estimate(method='trivial')
lip_fast_n = est_n.estimate(method='ECLipsE_Fast')
lip_acc_n = est_n.estimate(method='ECLipsE') 

# show results
print('For estimator created by nothing.')
print(f'Trivial Lipschitz estimate = {lip_trivial_n}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_n}')
print(f'Lipschitz estimate from EClipsE = {lip_acc_n}')

# relative tightness
print(f'Ratio Fast = {lip_fast_n / lip_trivial_n}')
print(f'Ratio Acc = {lip_acc_n / lip_trivial_n}')


