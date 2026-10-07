"""
Standalone Python script equivalent to Part_2_DeepONet_antiderivative_unaligned.ipynb.

Downloads the unaligned antiderivative dataset, trains a DeepXDE DeepONet via Neuromancer,
and saves plots and model evaluations.
"""

# %pip install "neuromancer[examples] @ git+https://github.com/pnnl/neuromancer.git@master"
# %pip install watermark

import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.callbacks import LossHistoryCallback
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules.operators import DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DATA_DIR = Path(__file__).resolve().parent / "data"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def fetch_and_load_npz(url: str, split_key: str):
    """Download NPZ if missing and return loaded arrays."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_DIR / f"antiderivative_unaligned_{split_key}.npz"

    if not filename.exists():
        print(f"Downloading {filename.name}...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as resp, open(filename, "wb") as f:
            f.write(resp.read())
        print("Done.")
    else:
        print(f"Found existing file: {filename}")

    with np.load(filename, allow_pickle=True) as d:
        return {
            "X": [d[f"X_{split_key}0"], d[f"X_{split_key}1"]],
            "y": d[f"y_{split_key}"],
        }


def prepare_data(dataset, name):
    """Prepare data for DeepONet training in Neuromancer using DeepXDE-style inputs."""
    branch_inputs = torch.from_numpy(dataset["X"][0]).float()  # (Nsamples, m)
    trunk_inputs = torch.from_numpy(dataset["X"][1]).float()  # (Nsamples, 1)
    outputs = torch.from_numpy(dataset["y"]).float()  # (Nsamples, 1)

    print(
        f"{name} dataset: samples = {branch_inputs.shape[0]}, sensors = {branch_inputs.shape[1]}"
    )

    return DictDataset(
        {
            "branch_inputs": branch_inputs,  # (Nsamples, m)
            "trunk_inputs": trunk_inputs,  # (Nsamples, 1)
            "outputs": outputs,  # (Nsamples, 1)
        },
        name=name,
    )


def visualize_samples(dataset):
    """Plot a few branch functions and the first trunk/output scatter."""
    functions = dataset["X"][0]
    m = functions.shape[1]
    sensors = np.linspace(0, 1, m)
    num_plots = 5

    for i in range(num_plots):
        plt.figure()
        plt.plot(sensors, functions[i], label=f"v (sample {i})")
        plt.scatter(sensors, functions[i], s=10, alpha=0.4, color="k")
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.title(f"Branch input function {i}")
        plt.legend()
        save_fig(f"sample_branch_{i}.png")

    x = dataset["X"][1].squeeze()
    u = dataset["y"].squeeze()
    plt.figure()
    plt.scatter(x[:5], u[:5], s=60)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.title("First five antiderivative evaluations")
    save_fig("sample_trunk_outputs.png")


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    os.environ["DDE_BACKEND"] = "pytorch"

    torch.manual_seed(1234)
    np.random.seed(1234)

    device = choose_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data download/split
    # -------------------------------------------------------------------------
    data_url = "https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/EnTn0aLimaRJuNKDOc0lfHkB2MXK8n8vAO1oV5cWVdJo3w?e=OLp80r"
    train_url = data_url + "%2Fantiderivative_unaligned_train.npz"
    test_url = data_url + "%2Fantiderivative_unaligned_test.npz"

    dataset_train = fetch_and_load_npz(train_url, "train")
    dataset_test_full = fetch_and_load_npz(test_url, "test")

    # Split the provided test set into dev and test
    dev_size = 20000
    total_test = dataset_test_full["X"][0].shape[0]
    if total_test < dev_size:
        raise ValueError(f"Requested dev_size {dev_size} > available {total_test}")

    dataset_dev = {
        "X": [
            dataset_test_full["X"][0][:dev_size],
            dataset_test_full["X"][1][:dev_size],
        ],
        "y": dataset_test_full["y"][:dev_size],
    }

    dataset_test = {
        "X": [
            dataset_test_full["X"][0][dev_size:],
            dataset_test_full["X"][1][dev_size:],
        ],
        "y": dataset_test_full["y"][dev_size:],
    }

    print(
        "Train shapes:",
        dataset_train["X"][0].shape,
        dataset_train["X"][1].shape,
        dataset_train["y"].shape,
    )
    print(
        "Dev shapes:",
        dataset_dev["X"][0].shape,
        dataset_dev["X"][1].shape,
        dataset_dev["y"].shape,
    )
    print(
        "Test shapes:",
        dataset_test["X"][0].shape,
        dataset_test["X"][1].shape,
        dataset_test["y"].shape,
    )

    visualize_samples(dataset_train)

    # -------------------------------------------------------------------------
    # Dataset and loaders
    # -------------------------------------------------------------------------
    train_datadict = prepare_data(dataset_train, "train")
    dev_datadict = prepare_data(dataset_dev, "dev")
    test_datadict = prepare_data(dataset_test, "test")

    print("Dimensions check:")
    print("branch_inputs:", train_datadict.datadict["branch_inputs"].shape)
    print("outputs:", train_datadict.datadict["outputs"].shape)
    print("trunk_inputs:", train_datadict.datadict["trunk_inputs"].shape)

    batch_size = 5000
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
    # Model
    # -------------------------------------------------------------------------
    m = train_datadict.datadict["branch_inputs"].shape[1]
    dim_x = 1
    dde_deeponet = dde.nn.DeepONet(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
    )

    dde_deeponet_wrapped = DeepXDEWrapper(model=dde_deeponet, is_cartesian=False)

    node_dde_deeponet = Node(
        dde_deeponet_wrapped,
        ["branch_inputs", "trunk_inputs"],
        ["g"],
        name="dde_DeepOnet",
    )
    print(node_dde_deeponet)

    var_y_est = variable("g")
    var_y_true = variable("outputs")

    var_loss = (var_y_est == var_y_true) ^ 2
    var_loss.name = "residual_loss"
    objectives = [var_loss]

    loss = PenaltyLoss(objectives, constraints=[])
    problem = Problem([node_dde_deeponet], loss=loss, grad_inference=True)
    problem.show()

    lr = 0.001
    epochs = 200
    epoch_verbose = 10
    warmup = 100
    patience = 0

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    loss_history_callback = LossHistoryCallback(
        plots_dir=PLOTS_DIR / "deeponet_unaligned", show=False
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

    best_outputs = trainer.test(best_model)
    problem.load_state_dict(best_model)

    train_loss_history = [
        l.detach().cpu().numpy() for l in trainer.loss_history["train"]
    ]
    dev_loss_history = [l.detach().cpu().numpy() for l in trainer.loss_history["dev"]]
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    print(f"Mean test loss: {mean_test_loss}")
    print(f"len(train_loss_history): {len(train_loss_history)}")
    print(f"len(dev_loss_history): {len(dev_loss_history)}")

    plt.semilogy(train_loss_history, label="Train loss")
    plt.semilogy(dev_loss_history, label="Dev loss")
    plt.scatter(
        len(train_loss_history),
        mean_test_loss,
        label="Mean test loss",
        c="red",
        marker="x",
    )
    plt.xlabel("# Epochs")
    plt.legend()
    save_fig("training_history_dde_unaligned.png")

    # -------------------------------------------------------------------------
    # Evaluation examples
    # -------------------------------------------------------------------------
    # Evaluate on a test function
    k = 211
    v_sample = test_datadict.datadict["branch_inputs"][k]  # (m,)
    m_eval = v_sample.shape[0]
    x_grid = torch.linspace(0, 1, steps=m_eval).unsqueeze(1)  # (m, 1)

    branch_batch = v_sample.unsqueeze(0).expand(m_eval, -1).to(device)  # (m, m)
    trunk_batch = x_grid.to(device)  # (m, 1)

    with torch.no_grad():
        preds = problem.predict(
            {"branch_inputs": branch_batch, "trunk_inputs": trunk_batch}
        )
    u_pred = preds["g"].squeeze(1).cpu().numpy()  # (m,)

    h = 1.0 / (m_eval - 1)
    u_true = torch.cumsum(v_sample * h, dim=0).cpu().numpy()  # (m,)
    v_np = v_sample.cpu().numpy()
    x_np = x_grid.squeeze(1).cpu().numpy()  # (m,)

    plt.plot(x_np, v_np, label="v(x) branch input")
    plt.plot(x_np, u_true, label="u(x) numeric integral", linestyle="--")
    plt.plot(x_np, u_pred, label="u_hat(x) model")
    plt.xlabel("x")
    plt.legend()
    save_fig("eval_test_function_dde_unaligned.png")

    # Evaluate on the function v(x) = x^2
    x_grid = torch.linspace(0, 1, steps=m_eval).unsqueeze(1)  # (m, 1)
    v_fn = torch.pow(x_grid.squeeze(1), 2).unsqueeze(0)  # (1, m)

    branch_batch = v_fn.expand(m_eval, -1).to(device)  # (m, m)
    trunk_batch = x_grid.to(device)  # (m, 1)

    with torch.no_grad():
        preds = problem.predict(
            {"branch_inputs": branch_batch, "trunk_inputs": trunk_batch}
        )
    u_pred = preds["g"].view(-1).cpu().numpy()  # (m,)

    x_np = x_grid.squeeze(1).cpu().numpy()
    v_np = v_fn.squeeze(0).cpu().numpy()
    u_true = (x_grid.squeeze(1) ** 3 / 3).cpu().numpy()  # analytic integral

    plt.plot(x_np, v_np, label="$v(x) = x^2$")
    plt.plot(x_np, u_true, label="$u(x) = x^3/3$ exact", linestyle="--")
    plt.plot(x_np, u_pred, label="$\\hat{u}(x)$ model")
    plt.xlabel("x")
    plt.legend()
    save_fig("eval_x2_dde_unaligned.png")

    # Evaluate on the function v(x) = cos(x)
    v_fn = torch.cos(x_grid.squeeze(1)).unsqueeze(0)  # (1, m)
    branch_batch = v_fn.expand(m_eval, -1).to(device)  # (m, m)
    trunk_batch = x_grid.to(device)  # (m, 1)

    with torch.no_grad():
        preds = problem.predict(
            {"branch_inputs": branch_batch, "trunk_inputs": trunk_batch}
        )
    u_pred = preds["g"].view(-1).cpu().numpy()  # (m,)

    x_np = x_grid.squeeze(1).cpu().numpy()
    v_np = v_fn.squeeze(0).cpu().numpy()
    u_true = torch.sin(x_grid.squeeze(1)).cpu().numpy()

    plt.plot(x_np, v_np, label="$v(x) = cos(x)$")
    plt.plot(x_np, u_true, label="$u(x) = sin(x)$ exact", linestyle="--")
    plt.plot(x_np, u_pred, label="$\\hat{u}(x)$ model")
    plt.xlabel("x")
    plt.legend()
    save_fig("eval_cos_dde_unaligned.png")


if __name__ == "__main__":
    main()
