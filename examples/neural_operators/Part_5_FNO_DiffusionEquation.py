"""
Standalone Python script equivalent to Part_5_FNO_DiffusionEquation.ipynb.

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

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.dataset import DictDataset
from neuromancer.modules.operators import FNO, H1Loss, LpLoss
from neuromancer.system import Node
from neuromancer.constraint import Loss
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuralop.utils import count_model_params


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


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


# exact solution y(x,t)
def f_real(x, t):
    return torch.exp(-t) * (torch.sin(np.pi * x))


def plot3d(X, T, y, name, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(T.numpy(), X.numpy(), y.numpy(), 20, cmap="viridis")
    fig.colorbar(cm, ax=ax1)  # Add a colorbar to a plot
    ax1.set_title("y(x,t)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_aspect("equal")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(T.numpy(), X.numpy(), y.numpy(), cmap="viridis")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    ax2.set_zlabel("y(x,t)")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(name)


def make_fno_grid(x_min, x_max, t_min, t_max, nx, nt, requires_grad=False):
    x = torch.linspace(x_min, x_max, nx)
    t = torch.linspace(t_min, t_max, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")
    Y = f_real(X, T)
    xt_grid = torch.stack([X, T], dim=0).unsqueeze(0).float()  # [1, 2, nx, nt]
    y_grid = Y.unsqueeze(0).unsqueeze(0).float()
    if requires_grad:
        xt_grid.requires_grad_(True)
    return xt_grid, y_grid


def main():
    warnings.filterwarnings("ignore")
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = select_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Define domain and exact solution
    # -------------------------------------------------------------------------
    x_min = -1
    x_max = 1
    t_min = 0
    t_max = 1

    total_points_x = 200
    total_points_t = 100

    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing="ij")
    y_real = f_real(X, T)

    print(x.shape, t.shape)
    print(X.shape, T.shape, y_real.shape)

    ymin = y_real.min()
    ymax = y_real.max()
    print(ymin, ymax)

    plot3d(X, T, y_real, "fno_diffusion_exact.png", "Exact solution y(x,t)")

    # -------------------------------------------------------------------------
    # Grid preparation
    # -------------------------------------------------------------------------
    train_nx, train_nt = 32, 32
    test_nx, test_nt = 16, 16

    XT_train_grid, Y_train_grid = make_fno_grid(x_min, x_max, t_min, t_max, train_nx, train_nt, requires_grad=True)
    XT_test_grid, Y_test_grid = make_fno_grid(x_min, x_max, t_min, t_max, test_nx, test_nt, requires_grad=False)

    # visualize collocation points for 2D input space (x, t)
    X_train_flat = XT_train_grid.squeeze(0)[0, :, :].flatten()
    T_train_flat = XT_train_grid.squeeze(0)[1, :, :].flatten()
    X_test_flat = XT_test_grid.squeeze(0)[0, :, :].flatten()
    T_test_flat = XT_test_grid.squeeze(0)[1, :, :].flatten()

    x_lb = torch.min(X_test_flat).item()
    x_ub = torch.max(X_test_flat).item()
    t_lb = torch.min(T_test_flat).item()
    t_ub = torch.max(T_test_flat).item()

    plt.figure()
    plt.scatter(
        X_train_flat.detach().numpy(),
        T_train_flat.detach().numpy(),
        s=4.0,
        c="blue",
        marker="o",
        label="Training grid",
    )
    plt.scatter(
        X_test_flat.detach().numpy(),
        T_test_flat.detach().numpy(),
        s=4.0,
        c="red",
        marker="o",
        label="Testing grid",
    )
    plt.title("(x,t) grids for FNO training")
    plt.xlim(x_lb, x_ub)
    plt.ylim(t_lb, t_ub)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend(loc="upper right")
    save_fig("fno_diffusion_grids.png")

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    XT_train_grid = XT_train_grid.to(device)
    Y_train_grid = Y_train_grid.to(device)

    train_ds_fno = DictDataset(
        {"xt_grid": XT_train_grid, "y_grid": Y_train_grid}, name="fno_train"
    )
    train_loader_fno = torch.utils.data.DataLoader(
        train_ds_fno, batch_size=1, collate_fn=train_ds_fno.collate_fn, shuffle=False
    )

    test_ds_fno = DictDataset(
        {"xt_grid": XT_test_grid, "y_grid": Y_test_grid}, name="fno_test"
    )
    test_loader_fno = torch.utils.data.DataLoader(
        test_ds_fno, batch_size=1, collate_fn=test_ds_fno.collate_fn, shuffle=False
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    fno_model = FNO(
        n_modes=(8, 8),
        in_channels=2,
        out_channels=1,
        hidden_channels=12,
        projection_channel_ratio=2,
    ).to(device)

    n_params = count_model_params(fno_model)
    print(f"\nOur model has {n_params} parameters.")
    sys.stdout.flush()

    fno_node = Node(fno_model, ["xt_grid"], ["y_hat_fno"], name="fno_model")
    print("symbolic inputs  of the fno_node:", fno_node.input_keys)
    print("symbolic outputs of the fno_node:", fno_node.output_keys)
    print(device)
    print(train_ds_fno.datadict["y_grid"].device)
    fno_model_out = fno_node(train_ds_fno.datadict)
    print(train_ds_fno.datadict["y_grid"].shape)
    print(fno_model_out["y_hat_fno"].shape)

    # -------------------------------------------------------------------------
    # Losses and problem
    # -------------------------------------------------------------------------
    h1_loss_fn = H1Loss(d=2)
    l2_loss_fn = LpLoss(d=2, p=2)

    h1_obj = Loss(
        ["y_hat_fno", "y_grid"],
        lambda yhat, y: h1_loss_fn(
            yhat.squeeze(1), y.squeeze(1)
        ),  # [B, 1, Nx, Nt] -> [B, Nx, Nt]
        name="h1_loss_fn",
    )

    l2_obj = Loss(
        ["y_hat_fno", "y_grid"],
        lambda yhat, y: l2_loss_fn(
            yhat.squeeze(1), y.squeeze(1)
        ),  # [B, 1, Nx, Nt] -> [B, Nx, Nt]
        name="l2_loss_fn",
    )

    objectives_fno = [h1_obj]  # L2 loss for training FNO
    constraints_fno = []
    loss_fno = PenaltyLoss(objectives_fno, constraints_fno)

    problem_fno = Problem(nodes=[fno_node], loss=loss_fno, grad_inference=True)
    problem_fno.show()

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(problem_fno.parameters(), lr=0.001)
    epochs = 500

    trainer = Trainer(
        problem_fno.to(device),
        train_loader_fno,
        optimizer=optimizer,
        epochs=epochs,
        epoch_verbose=10,
        train_metric="fno_train_loss",
        dev_metric="fno_train_loss",
        eval_metric="fno_train_loss",
        warmup=epochs,
        device=device,
    )

    start_time = time.time()
    best_model = trainer.train()
    print(f"Training wall time: {time.time() - start_time:.2f} seconds")

    problem_fno.load_state_dict(best_model)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    fno_out = problem_fno.nodes[0].cpu()
    y1 = fno_out(test_ds_fno.datadict)["y_hat_fno"]
    y_fno = y1.reshape(shape=[test_nx, test_nt]).detach().cpu()

    X_plot = XT_test_grid.squeeze(0)[0, :, :].detach().cpu()
    T_plot = XT_test_grid.squeeze(0)[1, :, :].detach().cpu()
    plot3d(X_plot, T_plot, y_fno, "fno_diffusion_pred.png", "FNO solution")

    Y_plot = Y_test_grid.squeeze(0).squeeze(0).detach().cpu()
    plot3d(X_plot, T_plot, Y_plot, "fno_diffusion_exact_test.png", "Exact PDE solution")

    plot3d(X_plot, T_plot, y_fno - Y_plot, "fno_diffusion_residual.png", "Residuals (FNO - exact)")


if __name__ == "__main__":
    main()
