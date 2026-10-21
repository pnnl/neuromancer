"""
Standalone Python script equivalent to Part_1_antiderivative_aligned.ipynb.

Mirrors the notebook workflow end-to-end, using both Neuromancer and DeepXDE DeepONets with plotting and training.
"""

# %pip install "neuromancer[examples] @ git+https://github.com/pnnl/neuromancer.git@master"
# %pip install watermark

import os
import sys
import time
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"

# For GUI windows instead of saved images, set an interactive backend before importing pyplot:
# import matplotlib; matplotlib.use("TkAgg")
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.callbacks import LossHistoryCallback
from neuromancer.dataset import DictDataset
from neuromancer.modules.blocks import MLP
from neuromancer.modules.operators import DeepONetCartesianProd, DeepXDEDataWrapper
from neuromancer.system import Node
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def generate_grf_data(space, m: int, dataset_size: int):
    """Generate dataset of GRF functions and their antiderivatives using Explicit Euler Method."""
    features_local = space.random(size=dataset_size)
    sensors_local = np.linspace(0, 1, num=m)[:, None]
    y_local = space.eval_batch(features_local, sensors_local)

    h_local = 1 / m
    # Integrate features using Explicit Euler Method
    # Initialise np.zeros((dataset_size, 1), populate rest using y values.
    # S0 = 0 works already as initial condition
    increments = np.concatenate([np.zeros((dataset_size, 1)), y_local[:, :-1]], axis=1)

    # running sum column‑wise to get antiderivative values
    # s[i + 1] = s[i] + h * yi[i]
    anti_y_local = np.cumsum(increments, axis=1) * h_local

    return {"X": [y_local, sensors_local], "y": anti_y_local}


def prepare_data(dataset, name):
    """
    Prepares data for DeepONet training in Neuromancer.
    Args:
        dataset (dict): A dictionary containing 'X' and 'y' numpy arrays.
            X[0]: branch inputs (functions) in shape (Nsamples, m)
            X[1]: trunk inputs (sensor locations) (m,1)
            y: outputs (antiderivatives) (Nsamples, m)
        name (str): Name of the dataset.
    Returns:
        data (DictDataset): Neuromancer DictDataset with branch and trunk inputs and outputs.
    """
    branch_inputs = torch.as_tensor(dataset["X"][0], dtype=torch.float32)  # (N, m)
    sensors_local = torch.as_tensor(dataset["X"][1], dtype=torch.float32)  # (m, 1)
    outputs = torch.as_tensor(dataset["y"], dtype=torch.float32)  # (N, m)

    # share the same grid for each batch entry without copying
    trunk_inputs = sensors_local.expand(branch_inputs.shape[0], -1, -1)

    print(
        f"{name} dataset: samples = {branch_inputs.shape[0]}, m = {branch_inputs.shape[1]}"
    )

    return DictDataset(
        {
            "branch_inputs": branch_inputs,
            "trunk_inputs": trunk_inputs,
            "outputs": outputs,
        },
        name=name,
    )


def main():
    # Random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Data generation and visualization of samples
    # -------------------------------------------------------------------------
    m = 100  # resolution of sampled functions u and their antiderivatives.
    space = dde.data.GRF(N=m, length_scale=1)  # GRF space with resolution m
    dataset_size = 150  # total dataset size
    features = space.random(
        size=dataset_size
    )  # sample dataset_size functions from the GRF
    h = 1 / m  # step size for Euler integrator
    sensors = np.linspace(0, 1, num=m)[:, None]  # m sensor locations between [0,1]

    # evaluate the sampled functions at the sensor locations
    y = space.eval_batch(features, sensors)
    print("Shape of sampled functions:", y.shape)  # (dataset_size, m)

    anti_y = []  # to store the integrated functions (antiderivatives)
    num_plots = 5  # number of functions to plot

    # Integrate features using Explicit Euler Method
    for idx, yi in enumerate(y):  # y has dataset_size=len(features)
        s = np.zeros(m)
        s0 = 0  # Initial Condition
        for i in range(0, m - 1):
            s[i + 1] = s[i] + h * yi[i]

        # Plotting routine for first num_plots samples
        if idx < num_plots:
            plt.figure()
            plt.plot(sensors, yi, "g", label=f"y_{idx}")
            plt.plot(sensors, s, "b", label=f"∫ y_{idx}")
            plt.scatter(sensors, yi, c="k", s=10, alpha=0.4, label="sensors")
            plt.legend(loc="lower right")
            save_fig(f"sample_{idx}.png")

        anti_y.append(s)

    anti_y = np.array(anti_y)
    print(f"Integrated {len(features)} samples; stored array shape: {anti_y.shape}")

    # check dimensions
    print("Dimensions check:")
    print("features:", features.shape)  # (dataset_size, m)
    print("y:", y.shape)  # (dataset_size, m)
    print("anti_y:", anti_y.shape)  # (dataset_size, m)
    print("sensors:", sensors.shape)  # (m, 1)

    # -------------------------------------------------------------------------
    # Dataset builders
    # -------------------------------------------------------------------------
    m = 100
    space = dde.data.GRF(N=m, length_scale=1)

    dataset_train = generate_grf_data(space, m, 150)
    dataset_dev = generate_grf_data(space, m, 50)
    dataset_test = generate_grf_data(space, m, 1000)

    # check dimensions
    print("Dimensions check:")
    print("y:", dataset_train["X"][0].shape)  # (dataset_size, m)
    print("anti_y:", dataset_train["y"].shape)  # (dataset_size, m)
    print("sensors:", dataset_train["X"][1].shape)  # (m, 1)

    train_datadict = prepare_data(dataset_train, name="train")
    dev_datadict = prepare_data(dataset_dev, name="dev")
    test_datadict = prepare_data(dataset_test, name="test")

    # check dimensions
    print("Dimensions check:")
    print("y:", train_datadict.datadict["branch_inputs"].shape)  # (dataset_size, m)
    print("anti_y:", train_datadict.datadict["outputs"].shape)  # (dataset_size, m)
    print(
        "sensors:", train_datadict.datadict["trunk_inputs"].shape
    )  # (dataset_size, m, 1), broadcasted

    batch_size = 100
    print(f"batch_size: {batch_size}")
    train_loader = torch.utils.data.DataLoader(
        train_datadict,
        batch_size=batch_size,
        collate_fn=train_datadict.collate_fn,
        shuffle=False,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_datadict,
        batch_size=batch_size,
        collate_fn=dev_datadict.collate_fn,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_datadict,
        batch_size=batch_size,
        collate_fn=test_datadict.collate_fn,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Model definition
    # -------------------------------------------------------------------------
    in_size_branch = m  # input size for branch net
    width_size = 40  # width of hidden layers
    depth_branch = 2  # depth of branch net
    interact_size = 40  # size of the interaction layer

    branch_net = MLP(
        insize=in_size_branch,
        outsize=interact_size,
        nonlin=nn.ReLU,
        hsizes=[width_size] * depth_branch,
        bias=True,
    )

    dim_x = 1  # dimension of the input x to the trunk net
    depth_trunk = 2  # depth of trunk net

    trunk_net = MLP(
        insize=dim_x,
        outsize=interact_size,
        nonlin=nn.ReLU,
        hsizes=[width_size] * depth_trunk,
        bias=True,
    )

    # Use existing DeepONet implementation
    deeponet = DeepONetCartesianProd(
        branch_net=branch_net, trunk_net=trunk_net, bias=True, return_transposed=False
    )

    node_deeponet = Node(
        deeponet, ["branch_inputs", "trunk_inputs"], ["g"], name="deeponet"
    )
    print(node_deeponet)

    # DeepONet from DeepXDE wrapped for Neuromancer DictDataset
    dde_deeponet = dde.nn.DeepONetCartesianProd(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
    )
    dde_deeponet_wrapped = DeepXDEDataWrapper(dde_deeponet)
    node_dde_deeponet = Node(
        dde_deeponet_wrapped,
        ["branch_inputs", "trunk_inputs"],
        ["g"],
        name="dde_deeponet",
    )
    print(node_dde_deeponet)

    var_y_est = variable("g")
    var_y_true = variable("outputs")

    var_loss = (var_y_est == var_y_true) ^ 2
    var_loss.name = "residual_loss"
    objectives = [var_loss]

    loss = PenaltyLoss(objectives, constraints=[])

    problem = Problem(nodes=[node_deeponet], loss=loss, grad_inference=True)
    problem.show()
    problem_dde = Problem(nodes=[node_dde_deeponet], loss=loss, grad_inference=True)
    problem_dde.show()

    lr = 0.001  # step size for gradient descent
    epochs = 10000  # number of training epochs
    epoch_verbose = (
        10  # print loss/display loss plot when this many epochs have occurred
    )
    warmup = 100  # number of epochs to wait before enacting early stopping policy
    patience = 0  # number of epochs with no improvement in eval metric to allow before early stopping

    # -------------------------------------------------------------------------
    # Trainer setup (Neuromancer DeepONet)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)

    loss_history_callback = LossHistoryCallback(
        plots_dir=PLOTS_DIR / "neuromancer", show=False
    )

    trainer = Trainer(
        problem.to(device),
        train_data=train_loader,
        dev_data=dev_loader,
        test_data=test_loader,
        optimizer=optimizer,
        logger=None,
        callback=loss_history_callback,
        epochs=epochs,
        patience=patience,
        epoch_verbose=epoch_verbose,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        warmup=warmup,
        device=device,
    )

    start_time = time.time()
    best_model = trainer.train()
    print(f"Training wall time: {time.time() - start_time:.2f} seconds")

    # load best trained model
    best_outputs = trainer.test(best_model)
    problem.load_state_dict(best_model)

    train_loss_history = [
        l.detach().cpu().numpy() for l in trainer.loss_history["train"]
    ]
    dev_loss_history = [l.detach().cpu().numpy() for l in trainer.loss_history["dev"]]
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    print(f"Neuromancer mean test loss: {mean_test_loss}")
    print(f"len(train_loss_history): {len(train_loss_history)}")
    print(f"len(dev_loss_history): {len(dev_loss_history)}")

    # -------------------------------------------------------------------------
    # Plot training history
    # -------------------------------------------------------------------------
    plt.semilogy(train_loss_history, label="Train loss (Neuromancer)")
    plt.semilogy(dev_loss_history, label="Dev loss (Neuromancer)")
    plt.scatter(
        len(train_loss_history),
        mean_test_loss,
        label="Mean test loss (Neuromancer)",
        c="red",
        marker="x",
    )
    plt.xlabel("# Epochs")
    plt.legend()
    save_fig("training_history_neuromancer.png")

    # -------------------------------------------------------------------------
    # DeepXDE DeepONet training and comparison
    # -------------------------------------------------------------------------
    optimizer_dde = torch.optim.AdamW(problem_dde.parameters(), lr=lr)
    loss_history_callback_dde = LossHistoryCallback(
        plots_dir=PLOTS_DIR / "deepxde", show=False
    )

    trainer_dde = Trainer(
        problem_dde.to(device),
        train_data=train_loader,
        dev_data=dev_loader,
        test_data=test_loader,
        optimizer=optimizer_dde,
        logger=None,
        callback=loss_history_callback_dde,
        epochs=epochs,
        patience=patience,
        epoch_verbose=epoch_verbose,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        warmup=warmup,
        device=device,
    )

    best_model_dde = trainer_dde.train()
    best_outputs_dde = trainer_dde.test(best_model_dde)
    problem_dde.load_state_dict(best_model_dde)

    train_loss_history_dde = [
        l.detach().cpu().numpy() for l in trainer_dde.loss_history["train"]
    ]
    dev_loss_history_dde = [
        l.detach().cpu().numpy() for l in trainer_dde.loss_history["dev"]
    ]
    mean_test_loss_dde = best_outputs_dde["mean_test_loss"].detach().cpu().numpy()
    print(f"DeepXDE mean test loss: {mean_test_loss_dde}")

    plt.semilogy(train_loss_history, label="Train loss (Neuromancer)")
    plt.semilogy(dev_loss_history, label="Dev loss (Neuromancer)")
    plt.semilogy(train_loss_history_dde, label="Train loss (DeepXDE)")
    plt.semilogy(dev_loss_history_dde, label="Dev loss (DeepXDE)")
    plt.scatter(
        len(train_loss_history_dde),
        mean_test_loss_dde,
        label="Mean test loss (DeepXDE)",
        c="orange",
        marker="x",
    )
    plt.xlabel("# Epochs")
    plt.legend()
    save_fig("training_history_comparison.png")

    # -------------------------------------------------------------------------
    # Evaluation examples
    # -------------------------------------------------------------------------
    # Evaluate on a test function
    k = 211  # k-th test function
    v_ = test_datadict.datadict["branch_inputs"][k : k + 1].to(device)  # (1, m)
    x_ = test_datadict.datadict["trunk_inputs"][k : k + 1].to(device)  # (1, m, 1)

    res = problem.predict({"branch_inputs": v_, "trunk_inputs": x_})

    u_true = test_datadict.datadict["outputs"][k].to(device)  # (m,)
    u_est = res["g"][0]  # (m,)
    grid = x_[0, :, 0].detach().cpu().numpy()

    plt.plot(grid, v_[0].detach().cpu().numpy(), label="v_")
    plt.plot(grid, u_true.detach().cpu().numpy(), label="u_")
    plt.plot(grid, u_est.detach().cpu().numpy(), label="u_est")
    plt.legend()
    save_fig("eval_test_function.png")

    # Evaluate on the function v(x) = x^2
    grid = train_datadict.datadict["trunk_inputs"][0]  # (m, 1)
    x_ = grid.unsqueeze(0).to(device)  # (1, m, 1)
    v_ = torch.pow(grid[:, 0], 2).unsqueeze(0).to(device)  # (1, m)

    res = problem.predict({"branch_inputs": v_, "trunk_inputs": x_})

    u_true = (1.0 / 3.0) * torch.pow(grid[:, 0], 3)  # (m,)
    u_est = res["g"][0]  # (m,)
    grid_np = grid[:, 0].detach().cpu().numpy()

    plt.plot(grid_np, v_[0].detach().cpu().numpy(), label="$v(x) = x^2$")
    plt.plot(grid_np, u_true.detach().cpu().numpy(), label="integral exact ($x^3/3$)")
    plt.plot(grid_np, u_est.detach().cpu().numpy(), label="integral estimated")
    plt.legend()
    save_fig("eval_x2.png")

    # Evaluate on the function v(x) = cos(x)
    grid = train_datadict.datadict["trunk_inputs"][0]  # (m, 1)
    x_ = grid.unsqueeze(0).to(device)  # (1, m, 1)
    v_ = torch.cos(grid[:, 0]).unsqueeze(0).to(device)  # (1, m)

    res = problem.predict({"branch_inputs": v_, "trunk_inputs": x_})

    u_true = torch.sin(grid[:, 0])  # (m,)
    u_est = res["g"][0]  # (m,)
    grid_np = grid[:, 0].detach().cpu().numpy()

    plt.plot(grid_np, v_[0].detach().cpu().numpy(), label="$v(x) = cos(x)$")
    plt.plot(grid_np, u_true.detach().cpu().numpy(), label="integral exact ($sin(x)$)")
    plt.plot(grid_np, u_est.detach().cpu().numpy(), label="integral estimated")
    plt.legend()
    save_fig("eval_cos.png")

    # Compare Neuromancer and DeepXDE models on one test sample
    k_compare = 211
    v_compare = test_datadict.datadict["branch_inputs"][k_compare : k_compare + 1].to(
        device
    )
    x_compare = test_datadict.datadict["trunk_inputs"][k_compare : k_compare + 1].to(
        device
    )
    res_base = problem.predict({"branch_inputs": v_compare, "trunk_inputs": x_compare})
    res_dde = problem_dde.predict(
        {"branch_inputs": v_compare, "trunk_inputs": x_compare}
    )

    u_true_compare = test_datadict.datadict["outputs"][k_compare].to(device)
    u_est_base = res_base["g"][0]
    u_est_dde = res_dde["g"][0]
    grid_compare = x_compare[0, :, 0].detach().cpu().numpy()

    plt.plot(grid_compare, v_compare[0].detach().cpu().numpy(), label="v_")
    plt.plot(grid_compare, u_true_compare.detach().cpu().numpy(), label="u_true")
    plt.plot(
        grid_compare, u_est_base.detach().cpu().numpy(), label="u_est (Neuromancer)"
    )
    plt.plot(grid_compare, u_est_dde.detach().cpu().numpy(), label="u_est (DeepXDE)")
    plt.legend()
    save_fig("eval_compare_neuromancer_vs_deepxde.png")


if __name__ == "__main__":
    main()
