import torch
import torch.nn as nn
import pytest
import neuromancer.slim as slim
from unittest import TestCase
from neuromancer.trainer import Trainer, LitTrainer
from neuromancer.problem import Problem, LitProblem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset, LitDataModule
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node




data_seed = 408  # random seed used for simulated data
torch.manual_seed(data_seed)
nsim = 5000  # number of datapoints: increase sample density for more robust results

# create dictionaries with sampled datapoints with uniform distribution
a_low, a_high, p_low, p_high = 0.2, 1.2, 0.5, 2.0

def data_setup_function(nsim=nsim, a_low=a_low, a_high=a_high, p_low=p_low, p_high=p_high): 

    
    samples_train = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                    "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_dev = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_test = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    # create named dictionary datasets
    train_data = DictDataset(samples_train, name='train')
    dev_data = DictDataset(samples_dev, name='dev')
    test_data = DictDataset(samples_test, name='test')

    batch_size = 64

    # Return the dict datasets in train, dev, test order, followed by batch_size 
    # If dev dataset is not wanted to be used, return None 
    return train_data, dev_data, test_data, batch_size 


def sample_problem(): 
    # define neural architecture for the trainable solution map
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=nn.ReLU,
                    hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class: sol_map(xi) -> x
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')
    # define decision variables
    x1 = variable("x")[:, [0]]
    x2 = variable("x")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable('p')
    a = variable('a')

    # objective function
    f = (1-x1)**2 + a*(x2-x1**2)**2
    obj = f.minimize(weight=1.0, name='obj')

    # constraints
    Q_con = 100.  # constraint penalty weights
    con_1 = Q_con*(x1 >= x2)
    con_2 = Q_con*((p/2)**2 <= x1**2+x2**2)
    con_3 = Q_con*(x1**2+x2**2 <= p**2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    # constrained optimization problem construction
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]

    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)

    return problem



@pytest.fixture(params=[sample_problem()])
def get_problem(request): 
    return request.param


@pytest.fixture(params=[data_setup_function])
def get_data(request): 
    return request.param

@pytest.fixture(params=[10])
def get_num_epochs(request): 
    return request.param

@pytest.fixture(params=[10])
def get_num_epochs(request): 
    return request.param








def compare_state_dicts(dict1, dict2):
    # Check if keys are the same
    if dict1.keys() != dict2.keys():
        return False

    # Check if values (tensors) are equal for each key
    for key in dict1.keys():
        if torch.equal(dict1[key], dict2[key]):
            return False
    return True



def test_trainer_initialization(get_problem, get_data): 

    epochs = 400
    patience = 11
    warmup = 100
    clip = 1.0 
    train_metric = 'dev_loss'
    dev_metric = 'train_loss'




    lit_data_module = LitDataModule(data_setup_function=get_data, nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    

    train_data, dev_data, test_data, batch_size = get_data(nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                        collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=test_data.collate_fn, shuffle=True)




    lit_trainer = LitTrainer(epochs=epochs, accelerator='cpu', train_metric=train_metric, dev_metric=dev_metric,clip=clip, \
                             warmup=warmup, patience=patience )
    base_trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=patience,
        epochs=epochs, 
        clip = clip, 
        train_metric=train_metric, 
        dev_metric=dev_metric

    )

    assert base_trainer.epochs == epochs 
    assert base_trainer.patience == patience 
    assert base_trainer.clip == clip 
    assert base_trainer.train_metric == train_metric 
    assert base_trainer.dev_metric == dev_metric 

    assert lit_trainer.epochs == epochs 
    assert lit_trainer.patience == patience 
    assert lit_trainer.clip == clip 
    assert lit_trainer.train_metric == train_metric 
    assert lit_trainer.dev_metric == dev_metric 

def test_train_runs_for_epochs(get_problem, get_data, get_num_epochs):
    
    lit_data_module = LitDataModule(data_setup_function=get_data, nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    

    train_data, dev_data, test_data, batch_size = get_data(nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                        collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=test_data.collate_fn, shuffle=True)


    num_epochs = get_num_epochs

    lit_trainer = LitTrainer(epochs=num_epochs, accelerator='cpu', save_weights=False)
    base_trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=99999,
        epochs=num_epochs
    )

    # Test for Standard PyTorch Trainer
    _ = base_trainer.train()
    assert base_trainer.current_epoch == num_epochs
    print("HELLO")
    # Test for PyTorch Lightning Trainer
    lit_trainer.fit(problem=get_problem, datamodule=lit_data_module)
    assert lit_trainer.current_epoch == num_epochs


def test_weight_updates(get_problem, get_data):
    problem = get_problem
    lit_data_module = LitDataModule(data_setup_function=get_data, nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    

    train_data, dev_data, test_data, batch_size = get_data(nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                        collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=test_data.collate_fn, shuffle=True)
    num_epochs = 10

    

    

    # Train for  1 epochs
    lit_trainer = LitTrainer(epochs=1, accelerator='cpu', save_weights=False)
    base_trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=99999,
        epochs=1
    )
    _ = base_trainer.train()
    lit_trainer.fit(problem=problem, datamodule=lit_data_module)

    lit_trainer_initial_weights = lit_trainer.get_weights().copy()
    print("lit trainer initial weights ", lit_trainer_initial_weights)
    base_trainer_initial_weights = base_trainer.best_model.copy()

    # Train for 9 more epochs
    lit_trainer = LitTrainer(epochs=9, accelerator='cpu', save_weights=False)
    base_trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=99999,
        epochs=9
    )
    _ = base_trainer.train()
    lit_trainer.fit(problem=problem, datamodule=lit_data_module)
    lit_trainer_final_weights = lit_trainer.get_weights().copy()
    base_trainer_final_weights = base_trainer.best_model.copy()

    compare_state_dicts(base_trainer_initial_weights, base_trainer_final_weights)
    compare_state_dicts(lit_trainer_initial_weights, lit_trainer_final_weights)

"""
def test_early_stopping(get_problem, get_data): 
    problem = get_problem
    lit_data_module = LitDataModule(data_setup_function=get_data, nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    

    train_data, dev_data, test_data, batch_size = get_data(nsim=nsim,a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                        collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                            collate_fn=test_data.collate_fn, shuffle=True)


    num_epochs = 400

    lit_trainer = LitTrainer(epochs=num_epochs, accelerator='cpu', patience=3, save_weights=False)
    base_trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=3,
        warmup=0,
        epochs=num_epochs
    )

    _ = base_trainer.train()
    lit_trainer.fit(problem=problem, datamodule=lit_data_module)

    assert base_trainer.current_epoch == 5 
    assert lit_trainer.current_epoch == 5
"""