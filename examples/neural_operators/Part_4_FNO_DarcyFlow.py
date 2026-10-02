"""
Standalone Python script equivalent to Part_4_FNO_DarcyFlow.ipynb.

Mirrors the notebook workflow end-to-end, including plotting and training.
"""

# %pip install neuromancer
# %pip install pyDOE

import warnings
import sys
from pathlib import Path
import os
import time

# For GUI windows instead of saved images, set an interactive backend before importing pyplot:
# import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from neuralop.data.datasets import DarcyDataset
from neuralop.layers.embeddings import GridEmbedding2D
from neuralop.utils import count_model_params

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.dataset import DictDataset
from neuromancer.modules.operators import FNO, H1Loss, LpLoss
from neuromancer.system import Node
from neuromancer.constraint import Loss, Objective, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stack_dataset(db, data_processor):
    """Stack and preprocess all samples from a neuralop dataset db."""
    xs, ys = [], []
    for i in range(len(db)):
        sample = data_processor.preprocess(db[i], batched=False)
        xs.append(sample["x"])
        ys.append(sample["y"].squeeze(0))  # [1, H, W]
    return torch.stack(xs), torch.stack(ys)


def make_dataloaders(dataset, device, res, batch_size_train=4, batch_size_test=4):
    """Create train and test loaders for a given resolution."""
    x_train, y_train = stack_dataset(dataset._train_db, dataset.data_processor)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_train.requires_grad_(True)
    y_train.requires_grad_(True)

    train_ds = DictDataset({"x_grid": x_train, "y_grid": y_train}, name="train")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )

    xs_t, ys_t = [], []
    for i in range(len(dataset.test_dbs[res])):
        sample = dataset.data_processor.preprocess(
            dataset.test_dbs[res][i], batched=False
        )
        xs_t.append(sample["x"])
        ys_t.append(sample["y"].squeeze(0))  # [1, H, W]
    x_test = torch.stack(xs_t)
    y_test = torch.stack(ys_t)

    test_ds = DictDataset({"x_grid": x_test, "y_grid": y_test}, name=f"test{res}")
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
    )
    return (
        train_loader,
        test_loader,
        train_ds,
        test_ds,
        x_train,
        y_train,
        x_test,
        y_test,
    )


def plot_embedding_sample(dataset, index=0):
    """Visualize one input sample with positional embeddings."""
    data = dataset.data_processor.preprocess(dataset._train_db[index], batched=False)
    positional_embedding = GridEmbedding2D(in_channels=1)
    x = positional_embedding(data["x"].unsqueeze(0)).squeeze(0)
    y = data["y"]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(x[0], cmap="gray")
    ax.set_title("Input x")
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(y.squeeze())
    ax.set_title("Output y")
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(x[1])
    ax.set_title("Positional embedding: x-coordinates")
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(x[2])
    ax.set_title("Positional embedding: y-coordinates")
    fig.suptitle("Visualizing one input sample with positional embeddings", y=0.98)
    save_fig("darcy_positional_embedding.png")


def plot_predictions(test_ds, preds, title, filename, max_samples=3):
    """Plot a few predictions vs ground truth."""
    fig = plt.figure(figsize=(7, 7))
    for index in range(min(max_samples, len(test_ds))):
        sample = test_ds[index]
        x = sample["x_grid"].cpu()
        y = sample["y_grid"].cpu()
        out = preds[index]

        ax = fig.add_subplot(3, 3, index * 3 + 1)
        ax.imshow(x[0], cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index * 3 + 2)
        ax.imshow(y.squeeze())
        if index == 0:
            ax.set_title("Ground-truth output")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index * 3 + 3)
        ax.imshow(out.squeeze().detach().numpy())
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle(title, y=0.98)
    save_fig(filename)


def main():
    warnings.filterwarnings("ignore")
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = select_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    darcy_dir = DATA_ROOT / "darcy_flow_small"
    darcy_dir.mkdir(parents=True, exist_ok=True)
    dataset = DarcyDataset(
        root_dir=str(darcy_dir),
        n_train=20,
        n_tests=[10, 10],
        batch_size=4,
        test_batch_sizes=[4, 2],
        train_resolution=16,
        test_resolutions=[16, 32],
        download=True,
    )
    print(dataset.__dict__.keys())

    train_db = dataset._train_db
    print(train_db.__dict__.keys())
    n_train = len(train_db)
    x_sample = train_db[0]["x"]
    res_x, res_y = x_sample.shape[-2:]
    print(f"length of training set = {n_train}")
    print(f"resolution = {res_x}")
    print(f"shape of grid = [{res_x}, {res_y}]")

    test_db = dataset._test_dbs[32]
    print(test_db.__dict__.keys())
    n_test = len(test_db)
    x_sample = test_db[0]["x"]
    res_x, res_y = x_sample.shape[-2:]
    print(f"length of training set = {n_test}")
    print(f"resolution = {res_x}")
    print(f"shape of grid = [{res_x}, {res_y}]")

    plot_embedding_sample(dataset, index=0)

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    res = 32
    batch_size_train = 4
    batch_size_test = 4
    (
        train_loader_fno,
        test_loader_fno,
        train_ds_fno,
        test_ds_fno,
        x_train,
        y_train,
        x_test,
        y_test,
    ) = make_dataloaders(
        dataset,
        device,
        res,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
    )

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    fno_model = FNO(
        n_modes=(8, 8),
        in_channels=1,
        out_channels=1,
        hidden_channels=12,
        projection_channel_ratio=2,
    ).to(device)

    n_params = count_model_params(fno_model)
    print(f"\nOur model has {n_params} parameters.")
    sys.stdout.flush()

    fno_node = Node(fno_model, ["x_grid"], ["y_fno"], name="fno_model")
    print("symbolic inputs  of the fno_node:", fno_node.input_keys)
    print("symbolic outputs of the fno_node:", fno_node.output_keys)
    print(train_ds_fno.datadict["y_grid"].device)
    print(f"Input training shape: {train_ds_fno.datadict['y_grid'].shape}")
    print(
        f"Forward Pass on Train data shape: {fno_node(train_ds_fno.datadict)['y_fno'].shape}"
    )

    # -------------------------------------------------------------------------
    # Losses and problem
    # -------------------------------------------------------------------------
    loss_approach = 3  # 1: H1Loss wrapper, 2: value/grad constraints, 3: Objective residuals (default)

    h1_loss_fn = H1Loss(d=2)
    l2_loss_fn = LpLoss(d=2, p=2)

    # Approach 1: H1 loss wrapper
    h1_obj = Loss(
        ["y_fno", "y_grid"],
        lambda yhat, y: h1_loss_fn(yhat.squeeze(1), y.squeeze(1)),
        name="h1_loss_fn",
    )

    l2_obj = Loss(
        ["y_fno", "y_grid"],
        lambda yhat, y: l2_loss_fn(yhat.squeeze(1), y.squeeze(1)),
        name="l2_loss_fn",
    )

    y_true = variable("y_grid")
    y_hat_fno = variable("y_fno")

    dx_hat = y_hat_fno[..., 1:, :] - y_hat_fno[..., :-1, :]
    dx_true = y_true[..., 1:, :] - y_true[..., :-1, :]

    dy_hat = y_hat_fno[..., :, 1:] - y_hat_fno[..., :, :-1]
    dy_true = y_true[..., :, 1:] - y_true[..., :, :-1]

    # Approach 2: constraint style
    val_constraint = (y_hat_fno == y_true) ^ 2
    val_constraint.update_name("h1_value")
    x_grad_constraint = (dx_hat == dx_true) ^ 2
    x_grad_constraint.update_name("h1_dx")
    y_grad_constraint = (dy_hat == dy_true) ^ 2
    y_grad_constraint.update_name("h1_dy")

    # Approach 3: Objective residuals
    residual_l2 = y_hat_fno - y_true
    l2_obj_residual = Objective(residual_l2**2, metric=torch.mean, name="l2_loss")
    residual_h1_dx = dx_hat - dx_true
    h1_dx_obj = Objective(residual_h1_dx**2, metric=torch.mean, name="h1_dx_loss")
    residual_h1_dy = dy_hat - dy_true
    h1_dy_obj = Objective(residual_h1_dy**2, metric=torch.mean, name="h1_dy_loss")

    if loss_approach == 1:
        objectives_fno = [h1_obj]
        constraints_fno = []
    elif loss_approach == 2:
        objectives_fno = [val_constraint, x_grad_constraint, y_grad_constraint]
        constraints_fno = []
    elif loss_approach == 3:
        objectives_fno = [l2_obj_residual, h1_dx_obj, h1_dy_obj]
        constraints_fno = []
    else:
        raise ValueError("loss_approach must be 1, 2, or 3.")

    loss_fno = PenaltyLoss(objectives_fno, constraints_fno)
    problem_fno = Problem(nodes=[fno_node], loss=loss_fno)
    problem_fno.show()

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(problem_fno.parameters(), lr=1e-2, weight_decay=1e-4)
    epochs = 50

    trainer = Trainer(
        problem_fno.to(device),
        train_loader_fno,
        optimizer=optimizer,
        epochs=epochs,
        epoch_verbose=10,
        train_metric="train_loss",
        dev_metric="train_loss",
        eval_metric="train_loss",
        warmup=epochs,
        device=device,
    )

    start_time = time.time()
    best_model = trainer.train()
    print(f"Training wall time: {time.time() - start_time:.2f} seconds")

    problem_fno.load_state_dict(best_model)

    # -------------------------------------------------------------------------
    # Evaluation (32x32)
    # -------------------------------------------------------------------------
    fno_eval = problem_fno.nodes[0].cpu()
    with torch.no_grad():
        preds = fno_eval(test_ds_fno.datadict)["y_fno"].cpu()
    plot_predictions(
        test_ds_fno,
        preds,
        "FNO predictions on test Darcy-Flow data",
        "fno_darcy_32.png",
    )

    # -------------------------------------------------------------------------
    # Evaluation (16x16)
    # -------------------------------------------------------------------------
    res = 16
    (
        _,
        test_loader_fno,
        _,
        test_ds_fno,
        _,
        _,
        x_test,
        y_test,
    ) = make_dataloaders(
        dataset,
        device,
        res,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
    )

    fno_eval = problem_fno.nodes[0].cpu()
    with torch.no_grad():
        preds = fno_eval(test_ds_fno.datadict)["y_fno"].cpu()
    plot_predictions(
        test_ds_fno,
        preds,
        "FNO predictions on test 16x16 Darcy-Flow data",
        "fno_darcy_16.png",
    )


if __name__ == "__main__":
    main()
