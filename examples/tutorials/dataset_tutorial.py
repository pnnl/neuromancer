"""
Datasets tutorial

This script demonstrates how to load datasets in neuromancer
Currently we support three types of datasets: static, sequence, and multi-sequence

"""


from neuromancer import dataset
import psl
import torch
import numpy as np


"""
Static dataset loading

suitable for classical ML tasks such as regression, classification
or for learning solutions to static constrained optimization problems
"""
# randomly sample two variables X and Y in train val and test set
train, val, test = [
    {"X": np.random.rand(400, 20), "Y": np.random.rand(400, 2)}
    for _ in range(3)
]
# normalize sampled datasets
train, stats = dataset.normalize_data(train, "zscore")
val, _ = dataset.normalize_data(val, "zscore", stats=stats)
test, _ = dataset.normalize_data(test, "zscore", stats=stats)
# create neuromancer static dataset
train_dset = dataset.StaticDataset(train, name="train")
val_dset = dataset.StaticDataset(val, name="val")
test_dset = dataset.StaticDataset(test, name="test")
# obtain full batch dictionary, keys = variable names, values = tensors of sampled data
train_fullbatch = train_dset.get_full_batch()
# get shapes of sampled variables
shapes = {k: v.shape for k, v in train_fullbatch.items() if isinstance(v, torch.Tensor)}
print(shapes)

"""
Sequence dataset loading

represents time series dataset with fixed sampling time
suitable for ODE system identification and ODE control tasks
"""
# loading arbitrary time series data from a CSV file
data_path = psl.datasets["aero"]
data = dataset.read_file(data_path)
# split sequence data with specified prediction horizon (nsteps)
train, val, test = dataset.split_sequence_data(data, nsteps=4)
# normalize sequence datasets
train, stats = dataset.normalize_data(train, "zscore")
val, _ = dataset.normalize_data(val, "zscore", stats=stats)
test, _ = dataset.normalize_data(test, "zscore", stats=stats)
# create neuromancer sequence dataset
train_dset = dataset.SequenceDataset(train, nsteps=4, name="train")
val_dset = dataset.SequenceDataset(val, nsteps=4, name="val")
test_dset = dataset.SequenceDataset(test, nsteps=4, name="test")
# get full batch and full sequence dictionaries
train_fullbatch = train_dset.get_full_batch()
train_fullseq = train_dset.get_full_sequence()
# get shapes of variables in the dataset
shapes_batch = {k: v.shape for k, v in train_fullbatch.items() if isinstance(v, torch.Tensor)}
shapes_seq = {k: v.shape for k, v in train_fullseq.items() if isinstance(v, torch.Tensor)}
print(shapes_batch)
print(shapes_seq)

"""
Multi-sequence dataset loading

represents multiple time series dataset with fixed sampling time
suitable for ODE system identification from multiple trajectories with different initial conditions
"""
# simulating multiple trajectories with random initial conditions
simulator = psl.emulators["TwoTank"](nsim=1024)
data = [
    simulator.simulate(x0=np.random.rand(2))
    for _ in range(15)
]
# split sequence data with specified prediction horizon (nsteps)
train, val, test = dataset.split_sequence_data(data, nsteps=16)
# normalize
train, stats = dataset.normalize_data(train, "zscore")
val, _ = dataset.normalize_data(val, "zscore", stats=stats)
test, _ = dataset.normalize_data(test, "zscore", stats=stats)
# create neuromancer sequence dataset
train_dset = dataset.SequenceDataset(train, nsteps=16)
val_dset = dataset.SequenceDataset(val, nsteps=16)
test_dset = dataset.SequenceDataset(test, nsteps=16)
# get full batch and full sequence dictionaries
train_fullbatch = train_dset.get_full_batch()
train_fullseq = train_dset.get_full_sequence()
# get shapes of variables in the dataset
shapes_batch = {k: v.shape for k, v in train_fullbatch.items() if isinstance(v, torch.Tensor)}
shapes_seq = {k: v.shape for k, v in train_fullseq[0].items() if isinstance(v, torch.Tensor)}
print(shapes_batch)
print(shapes_seq)