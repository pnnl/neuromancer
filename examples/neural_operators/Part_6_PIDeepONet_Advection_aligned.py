"""
Standalone Python script equivalent to Part_6_PIDeepONet_Advection_aligned.ipynb.

Mirrors the notebook workflow end-to-end, including plotting and training.
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

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules.operators import DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = select_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # PDE definition (advection: u_t + u_x = 0)
    # -------------------------------------------------------------------------
    def pde(x, y, v):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_t = dde.grad.jacobian(y, x, j=1)
        return dy_t + dy_x

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def func_ic(x, v):
        return v

    ic = dde.icbc.IC(geomtime, func_ic, lambda _, on_initial: on_initial)

    pde = dde.data.TimePDE(
        geomtime, pde, ic, num_domain=250, num_initial=50, num_test=400
    )

    n_bc = sum(pde.num_bcs)
    print(n_bc)
    print("train_x", pde.train_x.shape)
    print("train_x_all", pde.train_x_all.shape)
    print("train_x_bc", pde.train_x_bc.shape)
    print("test_x ", pde.test_x.shape)
    print(
        "BC x identical on test/train? ", np.allclose(pde.test_x[:n_bc], pde.train_x_bc)
    )

    plt.figure()
    plt.scatter(
        pde.train_x_all[:, 0], pde.train_x_all[:, 1], s=8, label="PDE (train_x_all)"
    )
    plt.scatter(
        pde.train_x_bc[:, 0], pde.train_x_bc[:, 1], s=18, label="BC (train_x_bc)"
    )
    plt.scatter(
        pde.test_x[:, 0], pde.test_x[:, 1], s=8, marker="x", label="Test (test_x)"
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Training & collocation points (domain & boundary)")
    plt.legend()
    save_fig("pideponet_advection_points.png")

    # -------------------------------------------------------------------------
    # GRF function space and sampled ICs
    # -------------------------------------------------------------------------
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
    xs = np.linspace(0, 1, 200)[:, None]
    feats = func_space.random(5)
    vals = func_space.eval_batch(feats, xs)
    plt.figure()
    for i in range(vals.shape[0]):
        plt.plot(xs.ravel(), vals[i])
    plt.title("Random IC samples from GRF")
    plt.xlabel("x")
    plt.ylabel("v(x)")
    save_fig("pideponet_grf_samples.png")

    # -------------------------------------------------------------------------
    # PDE operator dataset
    # -------------------------------------------------------------------------
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.data.PDEOperatorCartesianProd(
        pde,
        function_space=func_space,
        evaluation_points=eval_pts,
        num_function=1000,
        function_variables=[0],
        num_test=100,
        batch_size=32,
    )

    f_vals, x_trunk = data.train_x
    eval_pts = data.eval_pts
    f_trunk = data.train_aux_vars
    print("eval_pts.shape ", eval_pts.shape)
    print("x_trunk.shape ", x_trunk.shape)
    print("f_vals.shape    ", f_vals.shape)
    print("f_trunk.shape ", f_trunk.shape)

    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.plot(eval_pts[:, 0], f_vals[i], marker="o", label=f"v_{i}")
    plt.xlabel("x")
    plt.ylabel("v(x) at eval points")
    plt.legend()
    save_fig("pideponet_branch_samples.png")

    plt.figure()
    plt.plot(x_trunk[:, 0], x_trunk[:, 1], "rx", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Trunk points (interior + boundary)")
    save_fig("pideponet_trunk_points.png")

    (f_vals_test, x_trunk_test) = data.test_x
    f_trunk_test = data.test_aux_vars

    print("x_trunk_test.shape ", x_trunk_test.shape)
    print("f_vals_test.shape    ", f_vals_test.shape)
    print("f_trunk_test.shape ", f_trunk_test.shape)

    # -------------------------------------------------------------------------
    # Neuromancer datasets and loaders
    # -------------------------------------------------------------------------
    branch_inputs = torch.as_tensor(f_vals, dtype=torch.float32)
    trunk_grid = torch.as_tensor(x_trunk, dtype=torch.float32)
    trunk_inputs = trunk_grid.expand(branch_inputs.shape[0], -1, -1).requires_grad_()
    f_out_trunk = torch.as_tensor(f_trunk, dtype=torch.float32)

    train_datadict = DictDataset(
        {
            "branch_inputs": branch_inputs,
            "trunk_inputs": trunk_inputs,
            "outputs": f_out_trunk,
        },
        name="train",
    )

    branch_inputs_test = torch.as_tensor(f_vals_test, dtype=torch.float32)
    trunk_grid_test = torch.as_tensor(x_trunk_test, dtype=torch.float32)
    trunk_inputs_test = trunk_grid_test.expand(
        branch_inputs_test.shape[0], -1, -1
    ).requires_grad_()
    f_out_trunk_test = torch.as_tensor(f_trunk_test, dtype=torch.float32)

    test_datadict = DictDataset(
        {
            "branch_inputs": branch_inputs_test,
            "trunk_inputs": trunk_inputs_test,
            "outputs": f_out_trunk_test,
        },
        name="test",
    )
    dev_size = 20
    dev_idx = torch.arange(dev_size)
    test_idx = torch.arange(dev_size, len(test_datadict))

    dev_datadict = DictDataset(
        {k: v[dev_idx] for k, v in test_datadict.datadict.items()},
        name="dev",
    )
    test_datadict = DictDataset(
        {k: v[test_idx] for k, v in test_datadict.datadict.items()},
        name="test",
    )

    print("Dimensions check:")
    print("Train branch:", train_datadict.datadict["branch_inputs"].shape)
    print("Train output:", train_datadict.datadict["outputs"].shape)
    print("Train trunk:", train_datadict.datadict["trunk_inputs"].shape)
    print("Dev branch:", dev_datadict.datadict["branch_inputs"].shape)
    print("Dev output:", dev_datadict.datadict["outputs"].shape)
    print("Dev trunk:", dev_datadict.datadict["trunk_inputs"].shape)
    print("Test branch:", test_datadict.datadict["branch_inputs"].shape)
    print("Test output:", test_datadict.datadict["outputs"].shape)
    print("Test trunk:", test_datadict.datadict["trunk_inputs"].shape)

    batch_size = 32
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
    # DeepONet model (DeepXDE) wrapped for Neuromancer
    # -------------------------------------------------------------------------
    dim_x = 2
    eval_pts_shape = eval_pts.shape[0]
    dde_deeponet = dde.nn.DeepONetCartesianProd(
        [eval_pts_shape, 128, 128, 128],
        [dim_x, 128, 128, 128],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )

    dde_deeponet_wrapped = DeepXDEWrapper(model=dde_deeponet, is_cartesian=True)
    node_dde_deeponet = Node(
        dde_deeponet_wrapped,
        ["branch_inputs", "trunk_inputs"],
        ["g"],
        name="dde_deeponet",
    )
    print(node_dde_deeponet)
    print("symbolic inputs:", node_dde_deeponet.input_keys)
    print("symbolic outputs:", node_dde_deeponet.output_keys)

    net_out = node_dde_deeponet(train_datadict.datadict)
    print(net_out["g"].shape)

    # -------------------------------------------------------------------------
    # Physics-informed loss
    # -------------------------------------------------------------------------
    u = variable("g")
    xt = variable("trunk_inputs")
    v = variable("outputs")
    num_bc = sum(pde.num_bcs)

    du_dxt = u.grad(xt)
    u_x = du_dxt[..., 0]
    u_t = du_dxt[..., 1]

    print(du_dxt({**net_out, **train_datadict.datadict}).shape)
    print(u_x({**net_out, **train_datadict.datadict}).shape)
    print(u_t({**net_out, **train_datadict.datadict}).shape)
    print(v({**net_out, **train_datadict.datadict}).shape)

    res_pde = (u_t + u_x)[:, num_bc:]
    pde_loss = (res_pde == 0.0) ^ 2
    pde_loss.name = "pde_residual"
    res_pde.show()

    u_ic = u[:, :num_bc]
    v_ic = v[:, :num_bc]
    ic_loss = (u_ic == v_ic) ^ 2
    ic_loss.name = "ic"

    print(v_ic({**net_out, **train_datadict.datadict}).shape)
    print(u_ic({**net_out, **train_datadict.datadict}).shape)

    loss = PenaltyLoss(objectives=[pde_loss, ic_loss], constraints=[])

    problem = Problem(nodes=[node_dde_deeponet], loss=loss, grad_inference=True)
    problem.show()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    lr = 0.0005
    epochs = 2000
    epoch_verbose = 10
    warmup = 0
    patience = epochs

    optimizer = torch.optim.Adam(problem.parameters(), lr=lr)

    trainer = Trainer(
        problem.to(device),
        train_data=train_loader,
        dev_data=dev_loader,
        test_data=test_loader,
        optimizer=optimizer,
        logger=None,
        epochs=epochs,
        patience=patience,
        epoch_verbose=epoch_verbose,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="mean_train_loss",
        warmup=warmup,
        device=device,
    )

    start_time = time.time()
    best_model = trainer.train()
    print(f"Training wall time: {time.time() - start_time:.2f} seconds")

    problem.load_state_dict(best_model)
    best_outputs = trainer.test(best_model)

    train_loss_history = [
        l.detach().cpu().numpy() for l in trainer.loss_history["train"]
    ]
    print(f"len(train_loss_history): {len(train_loss_history)}")
    dev_loss_history = [l.detach().cpu().numpy() for l in trainer.loss_history["dev"]]
    print(f"len(dev_loss_history): {len(dev_loss_history)}")
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    print(mean_test_loss)

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
    save_fig("pideponet_training_history.png")

    # -------------------------------------------------------------------------
    # Post-processing on a regular grid
    # -------------------------------------------------------------------------
    problem.eval()

    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)
    xv, tv = np.meshgrid(x, t)

    u_true = np.sin(2 * np.pi * (xv - tv))
    v_branch_np = np.sin(2 * np.pi * eval_pts[:, 0])[None, :]

    x_trunk = np.stack([xv.ravel(), tv.ravel()], axis=-1)
    xt = torch.tensor(x_trunk, dtype=torch.float32, device=device).unsqueeze(0)
    v = torch.tensor(v_branch_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        res = problem.predict({"branch_inputs": v, "trunk_inputs": xt})
    u_pred = res["g"][0].detach().cpu().numpy().reshape(len(t), len(x))

    error = np.abs(u_true - u_pred)
    rel_l2 = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {rel_l2:.4e}")

    vmin = min(u_true.min(), u_pred.min())
    vmax = max(u_true.max(), u_pred.max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_params = {
        "origin": "lower",
        "aspect": "auto",
        "vmin": vmin,
        "vmax": vmax,
        "extent": [0, 1, 0, 1],
    }

    im1 = axes[0].imshow(u_true, **plot_params)
    axes[0].set_title("Analytic Solution u_true")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(u_pred, **plot_params)
    axes[1].set_title(f"DeepONet Prediction u_pred - L2 error {rel_l2:.4e}")
    axes[1].set_xlabel("x")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        error, origin="lower", aspect="auto", cmap="magma", extent=[0, 1, 0, 1]
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    save_fig("pideponet_solution_comparison.png")


if __name__ == "__main__":
    main()
