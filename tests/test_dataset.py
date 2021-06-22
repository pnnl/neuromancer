import random

from hypothesis import given, settings, strategies as st
from neuromancer import dataset
from neuromancer.blocks import Linear
from neuromancer.dynamics import (
    BlackSSM,
    BlockSSM,
)
from neuromancer.estimators import (
    LinearEstimator
)
from neuromancer.problem import Problem
import numpy as np
import torch
from torch import nn


def generate_test_data():
    test_data = [
        [random.random() for _ in range(20)]
        for _ in range(1000)
    ]
    header = [
        *[f"y{i}" for i in range(10)],
        *[f"u{i}" for i in range(5)],
        *[f"d{i}" for i in range(5)]
    ]
    return header, test_data


@given(
    st.integers(1, 64),
)
@settings(max_examples=64, deadline=None)
def test_nstep_samples(nsteps):
    data = dataset.read_file("test_data.csv")
    dset = dataset.SequenceDataset(data, nsteps)
    keys = {*{k + "p" for k in data.keys()}, *{k + "f" for k in data.keys()}}
    for sample in dset:
        assert all([v.shape[0] == nsteps for v in sample.values()])
        assert len(keys - set(sample.keys())) == 0
    

@given(
    st.integers(1, 128),
    st.sampled_from(["zero-one", "one-one", "zscore"]),
    st.integers(1, 33),
    st.integers(1, 33),
)
@settings(max_examples=500, deadline=None)
def test_default_load_pipeline(nsteps, norm_type, train_pct, val_pct):
    data = dataset.read_file("test_data.csv")
    try:
        nstep_data, loop_data, dims = dataset.get_sequence_dataloaders(
            data, nsteps, norm_type=norm_type, split_ratio=[train_pct, val_pct],
        )
    except AssertionError as e:
        print(f"caught assert: {e}")


@given(
    st.integers(1, 64),
)
@settings(max_examples=64, deadline=None)
def test_get_full_sequence(nsteps):
    data = dataset.read_file("test_data.csv")
    dset = dataset.SequenceDataset(data, nsteps)

    loop = dset.get_full_sequence()
    for k in data:
        kp, kf = f"{k}p", f"{k}f"
        # np.allclose used due to lossy cast from double to single float
        assert loop[kp].shape[0] == data[k].shape[0] - nsteps
        assert np.allclose(loop[kp].squeeze(1), data[k][:-nsteps])
        assert loop[kf].shape[0] == data[k].shape[0] - nsteps
        assert np.allclose(loop[kf].squeeze(1), data[k][nsteps:])
