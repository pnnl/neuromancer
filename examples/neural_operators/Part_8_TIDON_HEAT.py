"""
Standalone Python script equivalent to Part_8_TIDON_HEAT.ipynb.

Mirrors the notebook workflow end-to-end, including plotting and training.
"""

import os
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from neuromancer.callbacks import Callback
from neuromancer.constraint import Objective
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.loss import PenaltyLoss
from neuromancer.modules.operators import DeepXDEIntegratorWrapper
from neuromancer.modules.operators import DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.system import System
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str, tight_layout: bool = True):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def flatten_time(U_seq, F_seq, nsteps=1):
    """
    Flatten time dimension of trajectory data for n-step training samples.

    Args:
        U_seq (Tensor): Shape (S, T+1, N) - sequence of states.
        F_seq (Tensor): Shape (S, T, F) - sequence of forcings.
        nsteps (int): Number of steps ahead for the target state.

    Returns:
        u_t (Tensor): Shape (S*(T-nsteps+1), 1, N) - current state.
        f_t (Tensor): Shape (S*(T-nsteps+1), nsteps, F) - forcing sequence.
        u_next (Tensor): Shape (S*(T-nsteps+1), N) - target state after nsteps.
    """
    S, T_plus_1, N = U_seq.shape
    T = T_plus_1 - 1
    F = F_seq.shape[-1]

    if nsteps > T:
        raise ValueError(f"nsteps={nsteps} exceeds sequence length T={T}")

    u_t = U_seq[:, : T - nsteps + 1, :]  # (S, T-n+1, N)
    u_next = U_seq[:, nsteps : T + 1, :]  # (S, T-n+1, N)

    f_t_seq = [F_seq[:, i : T - nsteps + 1 + i, :] for i in range(nsteps)]
    f_t = torch.stack(f_t_seq, dim=2)  # (S, T-n+1, nsteps, F)

    # Flatten samples
    u_t = u_t.reshape(-1, 1, N)
    f_t = f_t.reshape(-1, nsteps, F)
    u_next = u_next.reshape(-1, N)

    return u_t, f_t, u_next


def build_dictdataset_flat(u_t, f_t, u_next, name, trunk_grid):
    trunk_inputs = trunk_grid.expand(u_t.shape[0], -1, -1)  # (S*T, 100, 1)
    return DictDataset(
        {
            "u_t": u_t.float(),  # (S*T, 1, 100)
            "f_t": f_t.float(),  # (S*T, 1, 4)
            "outputs": u_next.float(),  # (S*T, 100)
            "trunk_inputs": trunk_inputs.float(),
        },
        name=name,
    )


def exponential_decay(step: int, decay_rate: float, transition_steps: int) -> float:
    """Exponential learning-rate decay applied per optimizer step."""
    return decay_rate ** (step / transition_steps)


class StepLRSchedulerCallback(Callback):
    """
    Scheduler that steps every batch end, and logs loss every `log_every`.

    Args:
        scheduler: Learning rate scheduler to step.
        log_every: Frequency of logging loss (in number of steps). If 0, no logging.
        global_step: Internal counter for total steps taken.
        epoch_step: Counter for steps within the current epoch.
        step_indices: List of global step indices where loss was logged.
        step_losses: List of losses corresponding to the logged step indices.
    """

    def __init__(self, scheduler, log_every=1):
        super().__init__()
        self.scheduler = scheduler
        self.log_every = log_every
        self.global_step = 0
        self._last_epoch = None
        self.epoch_step = 0
        self.step_indices = []
        self.step_losses = []

    def end_batch(self, trainer, output):
        if self._last_epoch != trainer.current_epoch:
            self._last_epoch = trainer.current_epoch
            self.epoch_step = 0

        self.scheduler.step()

        if self.log_every and self.epoch_step % self.log_every == 0:
            loss = output.get(trainer.train_metric)
            if loss is not None:
                loss_val = loss.detach().cpu().item()
                print(
                    f"epoch: {trainer.current_epoch} step: {self.epoch_step} "
                    f"{trainer.train_metric}: {loss_val:.6g}"
                )
                self.step_indices.append(self.global_step)
                self.step_losses.append(loss_val)

        self.epoch_step += 1
        self.global_step += 1


def autoregressive_rollout(
    sample_id_value,
    dynamics_model,
    solutions,
    controls,
    device,
    rollout_steps=None,
):
    system = dynamics_model.to(device)
    system.eval()

    with torch.no_grad():
        u0 = solutions[sample_id_value, 0, :].float().to(device)
        true_states = solutions[sample_id_value, :, :].float().to(device)
        control_seq = controls[sample_id_value, :, :].float().to(device)

        if rollout_steps is None:
            rollout_steps = control_seq.shape[0]
        system.nsteps = rollout_steps  # change nsteps for full rollout

        batch = {
            "u_t": u0.unsqueeze(0).unsqueeze(1),  # (B, 1, N)
            "f_t": control_seq[:rollout_steps].unsqueeze(0),  # (B, T, 4)
        }
        out = system(batch)
        rollout_pred = out["u_t"].squeeze(0)  # (T+1, N)

        true_traj = true_states[: rollout_steps + 1]
        mse_loss = torch.mean((true_traj - rollout_pred) ** 2)
        rel_l2_error = torch.linalg.norm(true_traj - rollout_pred) / torch.linalg.norm(
            true_traj
        )

    return mse_loss, rel_l2_error, rollout_pred, true_traj


def plot_rollout_results(
    true_states,
    rollout_pred,
    x_cord,
    dt_value,
    sample_id_value,
):
    x_axis = x_cord.squeeze()
    T_plus_1 = true_states.shape[0]
    time_axis = np.linspace(0, dt_value * (T_plus_1 - 1), T_plus_1)
    error = np.abs(true_states - rollout_pred)

    vmin = float(min(true_states.min(), rollout_pred.min()))
    vmax = float(max(true_states.max(), rollout_pred.max()))

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        3,
        3,
        height_ratios=[20, 0.3, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.15,
        wspace=0.2,
        bottom=0.15,
        top=0.88,
        left=0.08,
        right=0.95,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)

    im0 = ax0.contourf(
        x_axis, time_axis, true_states, levels=120, cmap="inferno", vmin=vmin, vmax=vmax
    )
    ax0.set_title("FDM Prediction", fontsize=22)
    ax0.set_xlabel("x", fontsize=20)
    ax0.set_ylabel("t", fontsize=20)
    ax0.tick_params(labelsize=18)

    im1 = ax1.contourf(
        x_axis,
        time_axis,
        rollout_pred,
        levels=120,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("TI DON Prediction", fontsize=22)
    ax1.set_xlabel("x", fontsize=20)
    ax1.tick_params(labelsize=18, labelleft=False)

    im2 = ax2.contourf(x_axis, time_axis, error, levels=120, cmap="magma")
    ax2.set_title("Absolute Error", fontsize=22)
    ax2.set_xlabel("x", fontsize=20)
    ax2.tick_params(labelsize=18, labelleft=False)

    cbar_ax1 = fig.add_subplot(gs[2, 0:2])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label("u", fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    cbar_ax2 = fig.add_subplot(gs[2, 2])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label("Abs. Error", fontsize=20)
    cbar2.ax.tick_params(labelsize=18)

    save_fig(f"rollout_sample_{sample_id_value}.png", tight_layout=False)


def rollout_and_plot(
    sample_id_value,
    dynamics_model,
    solutions,
    controls,
    x_cord,
    dt,
    device,
    rollout_steps=None,
):
    mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
        sample_id_value,
        dynamics_model,
        solutions,
        controls,
        device,
        rollout_steps=rollout_steps,
    )

    plot_rollout_results(
        true_states.detach().cpu().numpy(),
        rollout_pred.detach().cpu().numpy(),
        x_cord.detach().cpu().numpy(),
        dt,
        sample_id_value,
    )

    return mse_loss, rel_l2_error


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    torch.set_default_dtype(torch.float)
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    data_path = "pathto/datasets/heat_smooth_f_dataset.npz"

    dataset = np.load(data_path)  # total 3000 samples available
    samples_to_load = 100  # samples or number of trajectories to load

    solutions = torch.from_numpy(
        dataset["solutions"][:samples_to_load]
    )  # shape: (samples, T+1, N)
    controls = torch.from_numpy(
        dataset["controls"][:samples_to_load]
    )  # shape: (samples, T, 4)
    x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)  # shape: (N, 1)
    dt = dataset["dt"]  # likely a scalar numpy value

    print(f"Loaded {samples_to_load}/{dataset['solutions'].shape[0]} samples:")
    print("solutions:", solutions.shape)
    print("controls:", controls.shape)
    print("x:", x_cord.shape)
    print("dt:", dt)

    # -------------------------------------------------------------------------
    # Sequences and split
    # -------------------------------------------------------------------------
    U = solutions  # (samples, T+1, N)
    F = controls  # (samples, T, 4)

    train_frac = 0.8
    n_train = int(train_frac * U.shape[0])

    U_train, U_test = U[:n_train], U[n_train:]
    F_train, F_test = F[:n_train], F[n_train:]

    # -------------------------------------------------------------------------
    # Flatten time for one-step training
    # -------------------------------------------------------------------------
    u_t_tr, f_t_tr, u_next_tr = flatten_time(U_train, F_train)
    u_t_te, f_t_te, u_next_te = flatten_time(U_test, F_test)

    # -------------------------------------------------------------------------
    # Neuromancer datasets
    # -------------------------------------------------------------------------
    trunk_grid = torch.as_tensor(x_cord, dtype=torch.float32).reshape(-1, 1)

    train_ds = build_dictdataset_flat(u_t_tr, f_t_tr, u_next_tr, "train", trunk_grid)
    test_ds = build_dictdataset_flat(u_t_te, f_t_te, u_next_te, "test", trunk_grid)

    print("Dimensions check train:")
    print("u_t (step):", train_ds.datadict["u_t"].shape)
    print("f_t (step):", train_ds.datadict["f_t"].shape)
    print("outputs (u_next):", train_ds.datadict["outputs"].shape)
    print("trunk_inputs:", train_ds.datadict["trunk_inputs"].shape)

    print("Dimensions check test:")
    print("u_t (step):", test_ds.datadict["u_t"].shape)
    print("f_t (step):", test_ds.datadict["f_t"].shape)
    print("outputs (u_next):", test_ds.datadict["outputs"].shape)
    print("trunk_inputs:", test_ds.datadict["trunk_inputs"].shape)

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    batch_size = 75
    print(f"batch_size: {batch_size}")

    g_device = torch.Generator(device=device).manual_seed(seed)

    steps_per_epoch = 1000
    num_samples = steps_per_epoch * batch_size

    g_device = torch.Generator(device=device).manual_seed(seed)

    sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,
        num_samples=num_samples,
        generator=g_device,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=train_ds.collate_fn,
        drop_last=True,
        shuffle=False,  # must be False when sampler is provided
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # MIONet configuration
    # -------------------------------------------------------------------------
    activation = {
        "branch1": "gelu",
        "branch2": "gelu",
        "trunk": "tanh",
    }

    hidden_dim = 100

    layer_sizes_branch1 = [100, 128, 128, 128, hidden_dim]  # u_t: (B,100) -> (B,100)
    layer_sizes_branch2 = [4, 32, 32, 32, hidden_dim]  # f_t: (B,4)   -> (B,100)
    layer_sizes_trunk = [1, 64, 64, 64, hidden_dim]  # x:   (100,1) -> (100,100

    net = dde.nn.MIONetCartesianProd(
        layer_sizes_branch1=layer_sizes_branch1,
        layer_sizes_branch2=layer_sizes_branch2,
        layer_sizes_trunk=layer_sizes_trunk,
        activation=activation,
        kernel_initializer="Glorot normal",
        trunk_last_activation=True,
        merge_operation="mul",
        layer_sizes_merger=None,
        layer_sizes_output_merger=None,
    )

    # -------------------------------------------------------------------------
    # Wrap with Neuromancer operators
    # -------------------------------------------------------------------------
    deeponet_wrapped = DeepXDEWrapper(
        model=net,
        is_cartesian=True,
        branch_keys=["u_t", "f_t"],
    )

    deeponet_integrator = DeepXDEIntegratorWrapper(
        model=deeponet_wrapped,
        trunk_inputs=trunk_grid,
    )

    fxRK4 = integrators.DiffEqIntegrator(deeponet_integrator, h=dt, method="rk4")

    node_rk4 = Node(
        fxRK4,
        ["u_t", "f_t"],
        ["u_t"],
        name="TI_DON + RK4",
    )

    # -------------------------------------------------------------------------
    # System for one-step training
    # -------------------------------------------------------------------------
    nsteps = 1
    dynamics_model = System(
        nodes=[node_rk4],
        name="Dynamics_system",
        nsteps=nsteps,
    )

    dynamics_model.show()

    # -------------------------------------------------------------------------
    # Dimension checks
    # -------------------------------------------------------------------------
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v), v)

    tensor_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "shape")}
    dynamics_model = dynamics_model.to(device)
    out = dynamics_model(tensor_batch)
    print("System out keys:", out.keys())
    print("u_t out shape:", out["u_t"].shape)

    # -------------------------------------------------------------------------
    # Loss and problem
    # -------------------------------------------------------------------------
    var_y_est = variable("u_t")[:, 1, :]
    var_y_true = variable("outputs")

    mse_var = (var_y_est - var_y_true) ** 2
    mse_obj = Objective(mse_var, metric=torch.mean, name="mse_loss")

    loss = PenaltyLoss(objectives=[mse_obj], constraints=[])

    problem = Problem(
        nodes=[dynamics_model],
        loss=loss,
        #   grad_inference=True
    )

    problem.show()

    # -------------------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------------------
    epochs = int(2)
    log_every = 100
    lr = 1e-3
    transition_steps = 2000
    decay_rate = 0.9

    optimizer = torch.optim.Adam(problem.parameters(), lr=lr)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: exponential_decay(step, decay_rate, transition_steps),
    )

    lr_callback = StepLRSchedulerCallback(scheduler, log_every=log_every)

    trainer = Trainer(
        problem.to(device),
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer,
        callback=lr_callback,
        epochs=epochs,
        epoch_verbose=1,
        train_metric="train_loss",
        dev_metric="train_loss",
        eval_metric="train_loss",
        test_metric="test_loss",
        warmup=epochs,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Train and evaluate
    # -------------------------------------------------------------------------
    best_model = trainer.train()

    trainer.dev_data = trainer.train_data
    best_outputs = trainer.test(best_model)
    problem.load_state_dict(best_model)

    train_loss_history = [
        l.detach().cpu().numpy() for l in trainer.loss_history["train"]
    ]
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    print(f"mean_test_loss: {mean_test_loss}")
    print(f"len(train_loss_history): {len(train_loss_history)}")

    # -------------------------------------------------------------------------
    # Plot training history
    # -------------------------------------------------------------------------
    epoch_steps = (np.arange(len(train_loss_history)) + 1) * steps_per_epoch

    plt.semilogy(
        lr_callback.step_indices, lr_callback.step_losses, label="Train loss (logged)"
    )
    plt.semilogy(epoch_steps, train_loss_history, "o-", label="Train loss (epoch)")

    plt.scatter(
        epoch_steps[-1],
        mean_test_loss,
        label="Mean test loss",
        c="red",
        marker="x",
    )

    plt.xlabel("# Steps")
    plt.legend()
    save_fig("training_history.png", tight_layout=False)

    # -------------------------------------------------------------------------
    # Autoregressive rollout (single sample)
    # -------------------------------------------------------------------------
    sample_id = 12
    mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
        sample_id,
        dynamics_model,
        solutions,
        controls,
        device,
    )
    print(mse_loss.item(), rel_l2_error.item())

    rollout_and_plot(
        sample_id,
        dynamics_model,
        solutions,
        controls,
        x_cord,
        dt,
        device,
    )

    # -------------------------------------------------------------------------
    # Evaluation over multiple samples
    # -------------------------------------------------------------------------
    num_samples_to_evaluate = 5
    dt_value = dt
    result_dir = None  # set to a path if you want to save figures

    sample_id = 0
    mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
        sample_id,
        dynamics_model,
        solutions,
        controls,
        device,
    )

    print(f"Test MSE Loss: {mse_loss.item():.4e}")
    print(f"Relative L2 Error: {rel_l2_error.item():.4e}")

    plot_rollout_results(
        true_states.detach().cpu().numpy(),
        rollout_pred.detach().cpu().numpy(),
        x_cord.detach().cpu().numpy(),
        dt_value,
        sample_id,
    )

    sample_indices = np.random.choice(
        solutions.shape[0], size=num_samples_to_evaluate, replace=False
    )

    all_losses = []
    all_l2_errors = []

    for idx in sample_indices:
        loss_val, l2_error, pred, true = autoregressive_rollout(
            idx,
            dynamics_model,
            solutions,
            controls,
            device,
        )
        all_losses.append(loss_val.item())
        all_l2_errors.append(l2_error.item())

        print(
            f"Sample {idx:4d} | MSE Loss: {loss_val.item():.4e} | Rel L2: {l2_error.item():.4e}"
        )

        plot_rollout_results(
            true.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
            x_cord.detach().cpu().numpy(),
            dt_value,
            idx,
        )

    print(f"\n==== Summary over {num_samples_to_evaluate} samples ====")
    print(f"Avg MSE Loss     : {np.mean(np.array(all_losses)):.4e}")
    print(f"Avg Rel L2 Error : {np.mean(np.array(all_l2_errors)):.4e}")


if __name__ == "__main__":
    main()
