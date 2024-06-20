import torch
from neuromancer.dataset import DictDataset
import numpy as np
def prepare_data(dataset, name):
    ## Note: transposing branch input because DictDataset in Neuromancer needs all tensors in the dict to have the same shape at index 0
    branch_inputs = dataset[f"X_{name}0"]
    trunk_inputs = dataset[f"X_{name}1"]
    outputs = dataset[f"y_{name}"]

    Nu = outputs.shape[0]
    Nsamples = outputs.shape[1]
    print(f'{name} dataset: Nu = {Nu}, Nsamples = {Nsamples}')

    # convert to pytorch tensors of float type
    t_branch_inputs = torch.from_numpy(branch_inputs).float()
    t_trunk_inputs = torch.from_numpy(trunk_inputs).float()
    t_outputs = torch.from_numpy(outputs).float()

    data = DictDataset({
        "branch_inputs": t_branch_inputs,
        "trunk_inputs": t_trunk_inputs,
        "outputs": t_outputs
    }, name=name)

    return data, Nu

def split_test_into_dev_test(original_test):
    dataset_dev = dict()
    dataset_test = dict()

    # split original test into dev and test
    dev_branch_inputs, test_branch_inputs = np.vsplit(original_test['X_test0'], 2)
    dev_trunk_inputs, test_trunk_inputs = np.vsplit(original_test['X_test1'], 2)
    dataset_dev['X_dev0'] = dev_branch_inputs
    dataset_dev['X_dev1'] = dev_trunk_inputs
    dataset_test['X_test0'] = test_branch_inputs
    dataset_test['X_test1'] = test_trunk_inputs
    dataset_dev['y_dev'], dataset_test['y_test'] = np.vsplit(original_test['y_test'], 2)
    return dataset_dev, dataset_test