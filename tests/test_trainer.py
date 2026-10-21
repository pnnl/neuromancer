import torch
import torch.nn as nn
import pytest
import neuromancer.slim as slim
from unittest import TestCase
from neuromancer.trainer import Trainer, LitTrainer
from neuromancer.callbacks import Callback
from neuromancer.problem import Problem, LitProblem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset, LitDataModule
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node


data_seed = 408  # random seed used for simulated data
torch.manual_seed(data_seed)
nsim = 500  # number of datapoints: increase sample density for more robust results

# create dictionaries with sampled datapoints with uniform distribution
a_low, a_high, p_low, p_high = 0.2, 1.2, 0.5, 2.0


def data_setup_function(
    nsim=nsim, a_low=a_low, a_high=a_high, p_low=p_low, p_high=p_high
):
    samples_train = {
        "a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
        "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
    }
    samples_dev = {
        "a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
        "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
    }
    samples_test = {
        "a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
        "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
    }
    # create named dictionary datasets
    train_data = DictDataset(samples_train, name="train")
    dev_data = DictDataset(samples_dev, name="dev")
    test_data = DictDataset(samples_test, name="test")

    batch_size = 64

    # Return the dict datasets in train, dev, test order, followed by batch_size
    # If dev dataset is not wanted to be used, return None
    return train_data, dev_data, test_data, batch_size


def sample_problem():
    # define neural architecture for the trainable solution map
    func = blocks.MLP(
        insize=2,
        outsize=2,
        bias=True,
        linear_map=slim.maps["linear"],
        nonlin=nn.ReLU,
        hsizes=[80] * 4,
    )
    # wrap neural net into symbolic representation of the solution map via the Node class: sol_map(xi) -> x
    sol_map = Node(func, ["a", "p"], ["x"], name="map")
    # define decision variables
    x1 = variable("x")[:, [0]]
    x2 = variable("x")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable("p")
    a = variable("a")

    # objective function
    f = (1 - x1) ** 2 + a * (x2 - x1**2) ** 2
    obj = f.minimize(weight=1.0, name="obj")

    # constraints
    Q_con = 100.0  # constraint penalty weights
    con_1 = Q_con * (x1 >= x2)
    con_2 = Q_con * ((p / 2) ** 2 <= x1**2 + x2**2)
    con_3 = Q_con * (x1**2 + x2**2 <= p**2)
    con_1.name = "c1"
    con_2.name = "c2"
    con_3.name = "c3"

    # constrained optimization problem construction
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]

    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)

    return problem


class RecordingCallback(Callback):
    """
    Helper class for testing Trainer callbacks.
    To check if the Trainer's evaluation hooks (begin_eval, end_eval)
    still run and don't crash with optional dev data.
    """

    def __init__(self):
        super().__init__()
        self.begin_eval_calls = 0
        self.end_eval_calls = 0
        self.last_output_keys = None

    def begin_eval(self, trainer, output):
        self.begin_eval_calls += 1
        self.last_output_keys = set(output.keys())

    def end_eval(self, trainer, output):
        self.end_eval_calls += 1


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
    epochs = 100
    patience = 11
    warmup = 100
    clip = 1.0
    train_metric = "dev_loss"
    dev_metric = "train_loss"

    lit_data_module = LitDataModule(
        data_setup_function=get_data,
        nsim=nsim,
        a_low=0.2,
        a_high=1.2,
        p_low=0.5,
        p_high=2.0,
    )

    train_data, dev_data, test_data, batch_size = get_data(
        nsim=nsim, a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dev_data.collate_fn,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=test_data.collate_fn,
        shuffle=True,
    )

    lit_trainer = LitTrainer(
        epochs=epochs,
        accelerator="cpu",
        train_metric=train_metric,
        dev_metric=dev_metric,
        clip=clip,
        warmup=warmup,
        patience=patience,
    )
    base_trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=patience,
        epochs=epochs,
        clip=clip,
        train_metric=train_metric,
        dev_metric=dev_metric,
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
    lit_data_module = LitDataModule(
        data_setup_function=get_data,
        nsim=nsim,
        a_low=0.2,
        a_high=1.2,
        p_low=0.5,
        p_high=2.0,
    )

    train_data, dev_data, test_data, batch_size = get_data(
        nsim=nsim, a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dev_data.collate_fn,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=test_data.collate_fn,
        shuffle=True,
    )

    num_epochs = get_num_epochs

    lit_trainer = LitTrainer(epochs=num_epochs, accelerator="cpu", save_weights=False)
    base_trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        test_loader,
        patience=99999,
        epochs=num_epochs,
    )

    # Test for Standard PyTorch Trainer
    _ = base_trainer.train()
    assert base_trainer.current_epoch == num_epochs
    # Test for PyTorch Lightning Trainer
    # lit_trainer.fit(problem=get_problem, datamodule=lit_data_module)
    lit_trainer.fit(problem=get_problem, data_setup_function=get_data)
    assert lit_trainer.current_epoch == num_epochs


def test_weight_updates(get_problem, get_data):
    problem = get_problem
    lit_data_module = LitDataModule(
        data_setup_function=get_data,
        nsim=nsim,
        a_low=0.2,
        a_high=1.2,
        p_low=0.5,
        p_high=2.0,
    )

    train_data, dev_data, test_data, batch_size = get_data(
        nsim=nsim, a_low=0.2, a_high=1.2, p_low=0.5, p_high=2.0
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dev_data.collate_fn,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=test_data.collate_fn,
        shuffle=True,
    )
    num_epochs = 10

    # Train for  1 epochs
    lit_trainer = LitTrainer(epochs=1, accelerator="cpu", save_weights=False)
    base_trainer = Trainer(
        problem, train_loader, dev_loader, test_loader, patience=99999, epochs=1
    )
    _ = base_trainer.train()
    lit_trainer.fit(problem=problem, data_setup_function=get_data)

    lit_trainer_initial_weights = lit_trainer.get_weights().copy()
    print("lit trainer initial weights ", lit_trainer_initial_weights)
    base_trainer_initial_weights = base_trainer.best_model.copy()

    # Train for 9 more epochs
    lit_trainer = LitTrainer(epochs=9, accelerator="cpu", save_weights=False)
    base_trainer = Trainer(
        problem, train_loader, dev_loader, test_loader, patience=99999, epochs=9
    )
    _ = base_trainer.train()
    lit_trainer.fit(problem=problem, data_setup_function=get_data)
    lit_trainer_final_weights = lit_trainer.get_weights().copy()
    base_trainer_final_weights = base_trainer.best_model.copy()

    compare_state_dicts(base_trainer_initial_weights, base_trainer_final_weights)
    compare_state_dicts(lit_trainer_initial_weights, lit_trainer_final_weights)


@pytest.mark.parametrize("include_dev", [True, False])
def test_trainer_optional_dev_data(get_problem, get_data, include_dev):
    """
    Parametrized test checks behavior with and without dev data.
    Make sure that callback runs without error in both cases
    """
    train_data, dev_data, _, batch_size = get_data(nsim=8)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    dev_loader = None
    if include_dev:
        dev_loader = torch.utils.data.DataLoader(
            dev_data,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=dev_data.collate_fn,
            shuffle=True,
        )

    callback = RecordingCallback()
    eval_metric = "dev_loss" if include_dev else "train_loss"

    trainer = Trainer(
        get_problem,
        train_loader,
        dev_loader,
        epochs=1,
        patience=1,
        eval_metric=eval_metric,
        callback=callback,
    )

    trainer.train()

    assert callback.begin_eval_calls == 1
    assert callback.end_eval_calls == 1
    if include_dev:
        assert f"mean_{trainer.dev_metric}" in callback.last_output_keys
        assert len(trainer.loss_history["dev"]) == 1
    else:
        assert f"mean_{trainer.train_metric}" in callback.last_output_keys
        assert trainer.loss_history["dev"] == []


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


def _train_with_grad_inference(get_data, epochs):
    """A short run where the dev loss carries an autograd graph, so storing it undetached would show."""
    problem = sample_problem()
    problem.grad_inference = True
    train_data, dev_data, _, batch_size = get_data()
    loaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0,
                                           collate_fn=d.collate_fn, shuffle=False)
               for d in [train_data, dev_data]]
    trainer = Trainer(problem, *loaders, epochs=epochs, patience=10, warmup=10,
                      epoch_verbose=100)
    trainer.train()
    return trainer


def test_stored_losses_do_not_retain_the_autograd_graph(get_data):
    """
    The loss history and best dev loss are kept for the whole run, so they must not hold
    graph-connected tensors: that pins one autograd graph, activations included, per epoch.
    """
    trainer = _train_with_grad_inference(get_data, epochs=3)

    assert len(trainer.loss_history["train"]) == 3
    assert len(trainer.loss_history["dev"]) == 3
    for split in ["train", "dev"]:
        for loss in trainer.loss_history[split]:
            assert loss.grad_fn is None
    assert trainer.best_devloss.grad_fn is None


def test_get_devloss_is_a_float(get_data):
    """get_devloss hands back a plain number whether or not any epoch has improved on the initial value."""
    train_data, dev_data, _, batch_size = get_data()
    loaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0,
                                           collate_fn=d.collate_fn, shuffle=False)
               for d in [train_data, dev_data]]
    trainer = Trainer(sample_problem(), *loaders, epochs=2, patience=10, warmup=10,
                      epoch_verbose=100, eval_metric="mean_dev_loss")

    # before training best_devloss is the numeric sentinel, not a tensor
    assert type(trainer.get_devloss()) is float
    assert trainer.get_devloss() == float(trainer.best_devloss)

    trainer.train()
    assert isinstance(trainer.best_devloss, torch.Tensor)
    assert type(trainer.get_devloss()) is float
    assert trainer.get_devloss() == trainer.best_devloss.item()
    assert trainer.get_devloss() == min(l.item() for l in trainer.loss_history["dev"])
