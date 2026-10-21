"""
Standalone Python script equivalent to Part_7_TIDON_DPC_HEAT.ipynb.

This version follows the notebook's current two-stage workflow:
1. train the TI-DeepONet dynamics model;
2. freeze the trained dynamics and train the DPC policy.
"""

import os
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from neuromancer.callbacks import Callback
from neuromancer.constraint import Objective, variable
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.modules.operators import DeepXDEIntegratorWrapper, DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node, System
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
    Flatten trajectory time dimension into n-step training samples.

    U_seq: (samples, T+1, N)
    F_seq: (samples, T, F)
    """
    samples, t_plus_1, state_dim = U_seq.shape
    horizon = t_plus_1 - 1
    control_dim = F_seq.shape[-1]

    if nsteps > horizon:
        raise ValueError(f"nsteps={nsteps} exceeds sequence length T={horizon}")

    u_t = U_seq[:, : horizon - nsteps + 1, :]
    u_next = U_seq[:, nsteps : horizon + 1, :]
    f_t_seq = [
        F_seq[:, i : horizon - nsteps + 1 + i, :] for i in range(nsteps)
    ]
    f_t = torch.stack(f_t_seq, dim=2)

    u_t = u_t.reshape(-1, 1, state_dim)
    f_t = f_t.reshape(-1, nsteps, control_dim)
    u_next = u_next.reshape(-1, state_dim)

    _ = samples
    return u_t, f_t, u_next


def build_dictdataset_flat(u_t, f_t, u_next, trunk_grid, name):
    trunk_inputs = trunk_grid.expand(u_t.shape[0], -1, -1)
    return DictDataset(
        {
            "u_t": u_t.float(),
            "f_t": f_t.float(),
            "outputs": u_next.float(),
            "trunk_inputs": trunk_inputs.float(),
        },
        name=name,
    )


def rbf_covariance_gpytorch(x_grid, lengthscale_eff, variance):
    """Compute RBF covariance using GPyTorch on CPU."""
    x = torch.tensor(x_grid, dtype=torch.float64).view(-1, 1)
    rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
    rbf_kernel.lengthscale = torch.tensor(float(lengthscale_eff), dtype=torch.float64)
    scale_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
    scale_kernel.outputscale = torch.tensor(float(variance), dtype=torch.float64)
    with torch.no_grad():
        cov = scale_kernel(x).evaluate()
    return cov.cpu().numpy().astype(np.float64)


def create_1d_grf_zero_boundary(x_grid, lengthscale, variance, kernel_type, seed):
    """Generate one 1D GRF sample with zero boundary conditions."""
    _ = kernel_type
    if seed is not None:
        np.random.seed(int(seed))

    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    n_points = x.shape[0]
    if n_points < 3:
        raise ValueError("x_grid must have at least 3 points for interior sampling.")

    boundary_indices = [0, n_points - 1]
    interior_indices = list(range(1, n_points - 1))

    lengthscale = float(lengthscale)
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive.")
    span = float(x[-1] - x[0])
    lengthscale_eff = lengthscale * span if span != 0.0 else lengthscale

    cov_full = rbf_covariance_gpytorch(x, lengthscale_eff, variance)
    cov_full = cov_full + 1e-10 * np.eye(n_points)

    k_ii = cov_full[np.ix_(interior_indices, interior_indices)]
    k_ib = cov_full[np.ix_(interior_indices, boundary_indices)]
    k_bb = cov_full[np.ix_(boundary_indices, boundary_indices)]
    k_bb_inv = np.linalg.inv(k_bb)
    k_conditional = k_ii - k_ib @ k_bb_inv @ k_ib.T
    k_conditional = 0.5 * (k_conditional + k_conditional.T)

    eigvals, eigvecs = np.linalg.eigh(k_conditional)
    eigvals = np.maximum(eigvals, 1e-10)
    l_mat = eigvecs @ np.diag(np.sqrt(eigvals))
    z = np.random.randn(len(interior_indices))
    interior_vals = l_mat @ z

    grf = np.zeros(n_points, dtype=np.float64)
    grf[interior_indices] = interior_vals
    return grf.astype(np.float32)


def generate_grf_pairs(
    num_samples,
    x_cord,
    length_scale,
    variance,
    num_modes,
    seed,
):
    """Generate GRF (u0, u_tf) pairs, no train/test split."""
    _ = num_modes
    rng = np.random.default_rng(None if seed is None else int(seed))
    if torch.is_tensor(x_cord):
        x_grid = x_cord.detach().cpu().numpy().astype(np.float32, copy=False)
        x_grid = x_grid.reshape(-1)
    else:
        x_grid = np.asarray(x_cord, dtype=np.float32).reshape(-1)

    grf_inits = []
    grf_finals = []
    for _ in range(int(num_samples)):
        s1 = int(rng.integers(0, 10_000))
        s2 = int(rng.integers(0, 10_000))
        grf_init = create_1d_grf_zero_boundary(
            x_grid, length_scale, variance, "rbf", seed=s1
        )
        grf_final = create_1d_grf_zero_boundary(x_grid, 0.4, variance, "rbf", seed=s2)
        grf_inits.append(grf_init)
        grf_finals.append(grf_final)

    inits = torch.from_numpy(np.array(grf_inits, dtype=np.float32))
    finals = torch.from_numpy(np.array(grf_finals, dtype=np.float32))
    return inits, finals


def build_dictdataset_policy(inits, finals, x_cord, nsteps, name="policy_grf"):
    """
    Build DPC policy dataset.

    inits:  (B, nx)
    finals: (B, nx)
    x_cord: (nx, 1)
    """
    if torch.is_tensor(x_cord):
        x_cord = x_cord.detach().cpu()
    else:
        x_cord = torch.as_tensor(x_cord, dtype=torch.float32)

    if x_cord.dim() == 1:
        x_cord = x_cord.reshape(-1, 1)

    trunk_inputs = x_cord.unsqueeze(0).expand(inits.shape[0], -1, -1)
    u_t_final = finals[:, None, :].repeat(1, nsteps + 1, 1)

    return DictDataset(
        {
            "u_t": inits[:, None, :].float(),
            "u_tf": u_t_final.float(),
            "trunk_inputs": trunk_inputs.float(),
        },
        name=name,
    )


class MLPControl(torch.nn.Module):
    """MLP control policy mapping (u_t, u_tf) to bounded controls."""

    def __init__(self, state_dim, control_dim, h_layers=None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        if h_layers is None:
            h_layers = [128, 128, 128, self.control_dim]
        self.h_layers = [int(v) for v in h_layers]
        self.output_dim = int(self.control_dim)
        if self.h_layers[-1] != self.output_dim:
            raise ValueError("Last h_layers entry must match output_dim.")

        input_dim = self.state_dim * 2
        layers = []
        prev_dim = input_dim
        for width in self.h_layers:
            layers.append(torch.nn.Linear(prev_dim, width))
            prev_dim = width
        self.layers = torch.nn.ModuleList(layers)
        self.output_bias = torch.nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, u_t, u_tf):
        if u_t.dim() != 2 or u_tf.dim() != 2:
            raise ValueError("u_t and u_tf must be 2D tensors.")
        if u_t.shape != u_tf.shape:
            raise ValueError("u_t and u_tf must have identical shapes.")
        if u_t.shape[1] != self.state_dim:
            raise ValueError("u_t second dimension must match state_dim.")

        inputs = torch.cat([u_t, u_tf], dim=1)
        x = inputs
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x))
        if x.shape[1] != self.output_dim:
            raise ValueError("Policy output dimension must match output_dim.")
        x = x + self.output_bias
        return torch.tanh(x) * 40.0


class PolicyMLPBounds(torch.nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.net = blocks.MLP_bounds(
            insize=2 * nx,
            outsize=nu,
            hsizes=[128, 128, 128],
            nonlin=activations["gelu"],
            min=-40.0,
            max=40.0,
        )

    def forward(self, u_t, u_tf):
        x = torch.cat([u_t, u_tf], dim=1)
        return self.net(x)


class StepLRSchedulerCallback(Callback):
    """Step a scheduler at each batch and optionally log batch loss."""

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
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"epoch: {trainer.current_epoch} step: {self.epoch_step} "
                    f"{trainer.train_metric}: {loss_val:.6g}"
                    f" lr: {lr:.2e}",
                    flush=True,
                )
                self.step_indices.append(self.global_step)
                self.step_losses.append(loss_val)

        self.epoch_step += 1
        self.global_step += 1


def exponential_decay_factory(decay_rate, transition_steps):
    def exponential_decay(step: int) -> float:
        return decay_rate ** (step / transition_steps)

    return exponential_decay


def predict_ct(u_t, u_tf, policy_node, key="u_tf"):
    out = policy_node({"u_t": u_t, key: u_tf})
    return out["f_t"]


def loss_test_fn_physical(
    policy_node,
    x_cord,
    test_inits,
    test_finals,
    rollout_steps=400,
    policy_key="u_tf",
):
    """Roll out a Crank-Nicolson PDE solver with Gaussian actuator forcing."""
    policy_net = policy_node.callable
    policy_device = next(policy_net.parameters()).device
    policy_dtype = next(policy_net.parameters()).dtype

    u_t0 = test_inits.detach()
    u_tf = test_finals.detach()
    if u_t0.dim() == 1:
        u_t0 = u_t0.unsqueeze(0)
    if u_tf.dim() == 1:
        u_tf = u_tf.unsqueeze(0)

    u_t0_cpu = u_t0.detach().cpu().to(dtype=torch.float64)
    u_tf_cpu = u_tf.detach().cpu().to(dtype=torch.float64)

    x = x_cord.detach().cpu().to(dtype=torch.float64).reshape(-1)
    n_points = int(x.shape[0])
    dx = 1.0 / n_points

    nu = 0.1
    sigma = 0.1
    centers = torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64)
    fixed_dt = 0.001

    r = nu * fixed_dt / (2 * dx**2)
    main_diag = torch.ones(n_points, device=x.device, dtype=torch.float64) * (1 + 2 * r)
    off_diag = torch.ones(n_points - 1, device=x.device, dtype=torch.float64) * (-r)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag[0] = 0.0
    if off_diag.numel() > 1:
        off_diag[-2] = 0.0

    A = torch.diag(main_diag)
    A = A + torch.diag(off_diag, diagonal=-1) + torch.diag(off_diag, diagonal=1)
    gaussian_basis = torch.exp(-0.5 * ((x[None, :] - centers[:, None]) / sigma) ** 2)

    def solve_step_implicit(u, f):
        rhs = u.clone()
        rhs[:, 1:-1] += r * (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2])
        rhs = rhs + fixed_dt * f
        rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
        return torch.linalg.solve(A, rhs.T).T

    traj_pred = []
    controls_pred = []
    u_t_cpu = u_t0_cpu
    u_t_policy = u_t0.to(device=policy_device, dtype=policy_dtype)
    u_tf_policy = u_tf.to(device=policy_device, dtype=policy_dtype)

    with torch.no_grad():
        for _ in range(int(rollout_steps)):
            c_t = predict_ct(u_t_policy, u_tf_policy, policy_node, key=policy_key)
            c_t_cpu = c_t.detach().to(device="cpu", dtype=torch.float64)
            f = c_t_cpu @ gaussian_basis
            u_next_cpu = solve_step_implicit(u_t_cpu, f)
            traj_pred.append(u_next_cpu.clone())
            controls_pred.append(c_t_cpu.clone())
            u_t_cpu = u_next_cpu
            u_t_policy = u_next_cpu.to(device=policy_device, dtype=policy_dtype)

    traj_pred = torch.stack(traj_pred, dim=0)
    controls_pred = torch.stack(controls_pred, dim=0)
    loss_pred = torch.mean((traj_pred[-1] - u_tf_cpu) ** 2)

    traj_zero = []
    u_t_cpu = u_t0_cpu
    with torch.no_grad():
        for _ in range(int(rollout_steps)):
            f = torch.zeros_like(u_t_cpu)
            u_next_cpu = solve_step_implicit(u_t_cpu, f)
            traj_zero.append(u_next_cpu.clone())
            u_t_cpu = u_next_cpu

    traj_zero = torch.stack(traj_zero, dim=0)
    return (
        traj_pred.to(dtype=torch.float32),
        traj_zero.to(dtype=torch.float32),
        controls_pred.to(dtype=torch.float32),
        u_t0_cpu.to(dtype=torch.float32),
        u_tf_cpu.to(dtype=torch.float32),
        loss_pred.to(dtype=torch.float32),
    )


def plot_rollout_comparison(
    x_cord,
    traj_pred,
    traj_zero,
    u_t0,
    u_tf,
    control_pred=None,
    step=20,
    save_path=None,
):
    """Plot rollout comparison panels for zero vs predicted control trajectories."""
    x = np.array(x_cord).squeeze()
    traj_pred = np.array(traj_pred)
    traj_zero = np.array(traj_zero)
    u_t0 = np.array(u_t0)
    u_tf = np.array(u_tf)

    t_pred, _ = traj_pred.shape
    t_zero = traj_zero.shape[0]

    plot_indices_pred = list(range(0, t_pred, int(step)))
    if (t_pred - 1) not in plot_indices_pred:
        plot_indices_pred.append(t_pred - 1)

    plot_indices_zero = list(range(0, t_zero, int(step)))
    if (t_zero - 1) not in plot_indices_zero:
        plot_indices_zero.append(t_zero - 1)

    cmap = plt.get_cmap("RdBu_r")
    _, axes = plt.subplots(1, 3, figsize=(22, 6))
    ax1, ax2, ax3 = axes

    label_fs = 20
    title_fs = 22
    tick_fs = 18
    legend_fs = 18

    for t in plot_indices_zero:
        color = cmap(t / (t_zero - 1))
        lw = 3 if t in [0, t_zero - 1] else 1.4
        alpha = 1.0 if t in [0, t_zero - 1] else 0.7
        ax1.plot(x, traj_zero[t], color=color, lw=lw, alpha=alpha)

    ax1.plot(x, u_t0, "k--", lw=2.5, label="Initial $u_{t0}$")
    ax1.plot(x, u_tf, "k:", lw=2.5, label="True $u_{tf}$")
    ax1.set_title("Evolution (Control = 0)", fontsize=title_fs)
    ax1.set_xlabel("x", fontsize=label_fs)
    ax1.set_ylabel("u(x)", fontsize=label_fs)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=legend_fs, loc="best")
    ax1.tick_params(axis="both", labelsize=tick_fs)

    for t in plot_indices_pred:
        color = cmap(t / (t_pred - 1))
        lw = 3 if t in [0, t_pred - 1] else 1.4
        alpha = 1.0 if t in [0, t_pred - 1] else 0.7
        ax2.plot(x, traj_pred[t], color=color, lw=lw, alpha=alpha)

    ax2.plot(x, u_t0, "k--", lw=2.5, label="Initial $u_{t0}$")
    ax2.plot(x, u_tf, "k:", lw=2.5, label="True $u_{tf}$")
    ax2.set_title("Evolution (Predicted Control)", fontsize=title_fs)
    ax2.set_xlabel("x", fontsize=label_fs)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=legend_fs, loc="best")
    ax2.tick_params(axis="both", labelsize=tick_fs)

    if control_pred is not None:
        control_pred = np.array(control_pred)
        t_ctrl, num_ctrl = control_pred.shape
        time = np.arange(t_ctrl)
        colors = plt.cm.tab10(np.linspace(0, 1, num_ctrl))
        for c in range(num_ctrl):
            ax3.plot(
                time,
                control_pred[:, c],
                lw=2.5,
                color=colors[c],
                label=f"f {c + 1}",
            )

        ax3.set_title("Predicted Control Signals", fontsize=title_fs, pad=15)
        ax3.set_xlabel("Time step", fontsize=label_fs)
        ax3.set_ylabel("Control amplitude", fontsize=label_fs, labelpad=10)
        ax3.grid(alpha=0.3)
        ax3.tick_params(axis="both", labelsize=tick_fs)
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax3.legend(
            fontsize=legend_fs, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1
        )
    else:
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "No control data provided",
            ha="center",
            va="center",
            fontsize=label_fs,
            color="gray",
        )

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if save_path is not None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")
    plt.close()


def make_tidon_model():
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde

    activation = {"branch1": "gelu", "branch2": "gelu", "trunk": "tanh"}
    hidden_dim = 100
    return dde.nn.MIONetCartesianProd(
        layer_sizes_branch1=[100, 128, 128, 128, hidden_dim],
        layer_sizes_branch2=[4, 32, 32, 32, hidden_dim],
        layer_sizes_trunk=[1, 64, 64, 64, hidden_dim],
        activation=activation,
        kernel_initializer="Glorot normal",
        trunk_last_activation=True,
        merge_operation="mul",
        layer_sizes_merger=None,
        layer_sizes_output_merger=None,
    )


def make_sampler(dataset, steps_per_epoch, batch_size, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=steps_per_epoch * batch_size,
        generator=generator,
    )


def main():
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

    # data_path = Path(
    #     os.environ.get("HEAT_DATA_PATH", "pathto/datasets/heat_smooth_f_dataset.npz")
    # )
    data_path = Path(
        os.environ.get("HEAT_DATA_PATH", "/home/pk222/projects/PDEControl_DPC/datasets/heat_smooth_f_dataset.npz")
    )
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Set HEAT_DATA_PATH to the .npz file."
        )

    dataset = np.load(data_path)
    n_data_load = 500
    solutions = torch.from_numpy(dataset["solutions"][:n_data_load])
    controls = torch.from_numpy(dataset["controls"][:n_data_load])
    x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)
    dt = float(np.asarray(dataset["dt"]).item())

    print(f"Loaded {n_data_load}/{dataset['solutions'].shape[0]} samples:")
    print("solutions:", solutions.shape)
    print("controls:", controls.shape)
    print("x:", x_cord.shape)
    print("dt:", dt)

    train_frac = 0.8
    U = solutions
    F = controls
    n_train = int(train_frac * U.shape[0])
    U_train, U_test = U[:n_train], U[n_train:]
    F_train, F_test = F[:n_train], F[n_train:]

    u_t_tr, f_t_tr, u_next_tr = flatten_time(U_train, F_train)
    u_t_te, f_t_te, u_next_te = flatten_time(U_test, F_test)

    trunk_grid = torch.as_tensor(x_cord, dtype=torch.float32).reshape(-1, 1)
    train_ds = build_dictdataset_flat(u_t_tr, f_t_tr, u_next_tr, trunk_grid, "train")
    test_ds = build_dictdataset_flat(u_t_te, f_t_te, u_next_te, trunk_grid, "test")

    print("TIDON dimensions check train:")
    print("u_t (step):", train_ds.datadict["u_t"].shape)
    print("f_t (step):", train_ds.datadict["f_t"].shape)
    print("outputs (u_next):", train_ds.datadict["outputs"].shape)
    print("trunk_inputs:", train_ds.datadict["trunk_inputs"].shape)

    print("TIDON dimensions check test:")
    print("u_t (step):", test_ds.datadict["u_t"].shape)
    print("f_t (step):", test_ds.datadict["f_t"].shape)
    print("outputs (u_next):", test_ds.datadict["outputs"].shape)
    print("trunk_inputs:", test_ds.datadict["trunk_inputs"].shape)

    batch_size_tidon = 75
    steps_per_epoch_tidon = 1000
    tidon_sampler = make_sampler(
        train_ds,
        steps_per_epoch_tidon,
        batch_size_tidon,
        seed,
        device,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size_tidon,
        sampler=tidon_sampler,
        collate_fn=train_ds.collate_fn,
        drop_last=True,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size_tidon,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )

    tidon = make_tidon_model()
    deeponet_wrapped = DeepXDEWrapper(
        model=tidon,
        is_cartesian=True,
        branch_keys=["u_t", "f_t"],
    )
    deeponet_integrator = DeepXDEIntegratorWrapper(
        model=deeponet_wrapped,
        trunk_inputs=trunk_grid,
    )

    fxRK4 = integrators.DiffEqIntegrator(deeponet_integrator, h=dt, method="rk4")
    node_rk4 = Node(fxRK4, ["u_t", "f_t"], ["u_t"], name="TI_DON + RK4")
    dynamics_model = System(nodes=[node_rk4], name="Dynamics_system", nsteps=1)

    var_y_est = variable("u_t")[:, 1, :]
    var_y_true = variable("outputs")
    mse_var = (var_y_est - var_y_true) ** 2
    mse_obj = Objective(mse_var, metric=torch.mean, name="mse_loss")
    tidon_loss = PenaltyLoss(objectives=[mse_obj], constraints=[])
    problem_tidon = Problem(nodes=[dynamics_model], loss=tidon_loss).to(device)

    epochs_tidon = 10
    log_every_tidon = 100
    lr = 1e-3
    transition_steps = 2000
    decay_rate = 0.9
    exponential_decay = exponential_decay_factory(decay_rate, transition_steps)

    optimizer_tidon = torch.optim.Adam(problem_tidon.parameters(), lr=lr)
    scheduler_tidon = LambdaLR(optimizer_tidon, lr_lambda=exponential_decay)
    lr_callback_tidon = StepLRSchedulerCallback(
        scheduler_tidon,
        log_every=log_every_tidon,
    )

    trainer_tidon = Trainer(
        problem_tidon,
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer_tidon,
        callback=lr_callback_tidon,
        epochs=epochs_tidon,
        epoch_verbose=1,
        train_metric="train_loss",
        dev_metric="train_loss",
        eval_metric="train_loss",
        test_metric="test_loss",
        device=device,
    )

    best_tidon_model = trainer_tidon.train()

    problem_tidon.load_state_dict(best_tidon_model)
    tidon.eval()
    for p in tidon.parameters():
        p.requires_grad_(False)
    print(
        "model params require grad after freeze:",
        any(p.requires_grad for p in tidon.parameters()),
    )

    nsteps = 100
    length_scale = 0.2
    variance = 1.0
    num_modes = 64
    num_samples_policy = 100

    inits, finals = generate_grf_pairs(
        num_samples=num_samples_policy,
        x_cord=trunk_grid,
        length_scale=length_scale,
        variance=variance,
        num_modes=num_modes,
        seed=seed,
    )

    num_train_policy = int(num_samples_policy * train_frac)
    train_idx = torch.arange(0, num_train_policy, device="cpu")
    test_idx = torch.arange(num_train_policy, num_samples_policy, device="cpu")

    train_ds_policy = build_dictdataset_policy(
        inits[train_idx],
        finals[train_idx],
        x_cord=trunk_grid,
        nsteps=nsteps,
        name="train",
    )
    test_ds_policy = build_dictdataset_policy(
        inits[test_idx],
        finals[test_idx],
        x_cord=trunk_grid,
        nsteps=nsteps,
        name="test",
    )

    print("DPC dimensions check train:")
    print("u_t (step):", train_ds_policy.datadict["u_t"].shape)
    print("u_tf (final state):", train_ds_policy.datadict["u_tf"].shape)
    print("trunk_inputs:", train_ds_policy.datadict["trunk_inputs"].shape)

    print("DPC dimensions check test:")
    print("u_t (step):", test_ds_policy.datadict["u_t"].shape)
    print("u_tf (final state):", test_ds_policy.datadict["u_tf"].shape)
    print("trunk_inputs:", test_ds_policy.datadict["trunk_inputs"].shape)

    batch_size_dpc = 75
    steps_per_epoch_dpc = 100
    dpc_sampler = make_sampler(
        train_ds_policy,
        steps_per_epoch_dpc,
        batch_size_dpc,
        seed,
        device,
    )
    train_loader_policy = torch.utils.data.DataLoader(
        train_ds_policy,
        batch_size=batch_size_dpc,
        sampler=dpc_sampler,
        collate_fn=train_ds_policy.collate_fn,
        drop_last=True,
        shuffle=False,
    )
    test_loader_policy = torch.utils.data.DataLoader(
        test_ds_policy,
        batch_size=batch_size_dpc,
        collate_fn=test_ds_policy.collate_fn,
        shuffle=False,
    )

    nx = 100
    nu = 4
    policyMLP = MLPControl(state_dim=nx, control_dim=nu).to(device=device)
    policy = Node(policyMLP, ["u_t", "u_tf"], ["f_t"], name="policy")

    fxRK4_frozen = integrators.RK4(deeponet_integrator, h=dt)
    node_rk4_frozen = Node(
        fxRK4_frozen,
        ["u_t", "f_t"],
        ["u_t"],
        name="TI_DON_frozen + RK4",
    )

    system_dpc = System([policy, node_rk4_frozen], nsteps=nsteps).to(device)

    batch = next(iter(train_loader_policy))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v), v)

    tensor_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "shape")}
    assert tensor_batch["u_tf"].shape[1] >= system_dpc.nsteps + 1
    out = system_dpc(tensor_batch)
    print("System out keys:", out.keys())
    print("f_t out shape:", out["f_t"].shape)
    print("u_t out shape:", out["u_t"].shape)
    print("u_tf out shape:", out["u_tf"].shape)

    y = variable("u_t")[:, -1, :]
    r = variable("u_tf")[:, -1, :]
    track = (y - r) ** 2
    mse_obj = Objective(track, metric=torch.mean, name="mse_loss")
    dpc_loss = PenaltyLoss(objectives=[mse_obj], constraints=[])
    problem = Problem([system_dpc], dpc_loss)

    epochs_dpc = 10
    log_every_dpc = 10
    dpc_trainable_params = [p for p in problem.parameters() if p.requires_grad]
    if not dpc_trainable_params:
        raise RuntimeError("No trainable DPC parameters found.")

    optimizer_dpc = torch.optim.Adam(dpc_trainable_params, lr=lr)
    scheduler_dpc = LambdaLR(optimizer_dpc, lr_lambda=exponential_decay)
    lr_callback_dpc = StepLRSchedulerCallback(
        scheduler_dpc,
        log_every=log_every_dpc,
    )

    trainer = Trainer(
        problem.to(device),
        train_data=train_loader_policy,
        test_data=test_loader_policy,
        optimizer=optimizer_dpc,
        callback=lr_callback_dpc,
        epochs=epochs_dpc,
        epoch_verbose=1,
        train_metric="train_loss",
        dev_metric="train_loss",
        eval_metric="train_loss",
        test_metric="test_loss",
        warmup=epochs_dpc,
        device=device,
    )

    print(
        "policy params require grad:",
        any(p.requires_grad for p in policyMLP.parameters()),
    )
    print("model params require grad:", any(p.requires_grad for p in tidon.parameters()))

    batch = next(iter(train_loader_policy))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(batch.keys())

    out = problem(batch)
    print(out.keys())
    print("loss requires grad:", out["train_loss"].requires_grad)

    best_dpc_model = trainer.train()

    trainer.dev_data = trainer.train_data
    problem.load_state_dict(best_dpc_model)

    train_loss_history = [
        loss.detach().cpu().numpy() for loss in trainer.loss_history["train"]
    ]
    print(f"len(train_loss_history): {len(train_loss_history)}")

    epoch_steps = (np.arange(len(train_loss_history)) + 1) * steps_per_epoch_dpc
    plt.semilogy(
        lr_callback_dpc.step_indices,
        lr_callback_dpc.step_losses,
        label="DPC train loss (logged)",
    )
    plt.semilogy(epoch_steps, train_loss_history, "o-", label="Train loss (epoch)")
    plt.xlabel("# Steps")
    plt.legend()
    save_fig("part_7_dpc_training_loss.png", tight_layout=False)

    num_eval_samples = 5
    rollout_steps = nsteps
    test_inits = inits[test_idx]
    test_finals = finals[test_idx]

    system_dpc = system_dpc.to(device)
    system_dpc.nsteps = nsteps

    u_t0 = test_inits[:, None, :].to(device)
    u_tf_seq = test_finals[:, None, :].repeat(1, nsteps + 1, 1).to(device)
    test_data = {"u_t": u_t0, "u_tf": u_tf_seq}

    trajectories = system_dpc(test_data)
    print(trajectories.keys())
    print(trajectories["u_t"].shape)

    with torch.no_grad():
        num_eval = min(num_eval_samples, test_inits.shape[0])
        eval_indices = torch.randperm(test_inits.shape[0], device="cpu")[:num_eval]
        eval_indices = eval_indices.tolist()

        for idx in eval_indices:
            print(f"\n=== Plotting sample {idx} ===")
            u0 = test_inits[idx : idx + 1].to(device)
            u_tf = test_finals[idx : idx + 1].to(device)

            u_t0 = u0[:, None, :]
            u_tf_seq = u_tf[:, None, :].repeat(1, rollout_steps + 1, 1)
            test_data = {"u_t": u_t0, "u_tf": u_tf_seq}

            trajectories = system_dpc(test_data)
            traj_pred = trajectories["u_t"]
            controls_pred = trajectories.get("f_t")

            u_t = u0
            traj_zero = []
            for _ in range(int(rollout_steps)):
                c_t = torch.zeros_like(controls_pred[:, 0, :])
                u_t = node_rk4_frozen.callable(u_t, c_t)
                traj_zero.append(u_t)
            traj_zero = torch.stack(traj_zero, dim=0)
            traj_zero = torch.cat([u0[None, ...], traj_zero], dim=0)

            loss_extended = torch.mean((traj_pred[:, -1, :] - u_tf) ** 2)

            (
                traj_pred_phys,
                traj_zero_phys,
                controls_pred_phys,
                u_t0_phys,
                u_tf_phys,
                loss_physical,
            ) = loss_test_fn_physical(
                policy,
                x_cord,
                u0,
                u_tf,
                rollout_steps=rollout_steps,
            )

            if not torch.isclose(u_t0_phys[0, 0], torch.tensor(0.0), atol=1e-6):
                print("Warning: u_t0 boundary at x=0 is not zero (physical).")
            if not torch.isclose(u_t0_phys[0, -1], torch.tensor(0.0), atol=1e-6):
                print("Warning: u_t0 boundary at x=1 is not zero (physical).")
            if not torch.isclose(
                traj_pred_phys[-1, 0, 0], torch.tensor(0.0), atol=1e-6
            ):
                print("Warning: physical rollout boundary at x=0 is not zero.")
            if not torch.isclose(
                traj_pred_phys[-1, 0, -1], torch.tensor(0.0), atol=1e-6
            ):
                print("Warning: physical rollout boundary at x=1 is not zero.")

            print(
                "Shapes:",
                tuple(traj_pred.shape),
                tuple(traj_zero.shape),
                tuple(controls_pred.shape),
                tuple(u_t0[0].shape),
                tuple(u_tf.shape),
                "Loss (extended):",
                float(loss_extended),
                "Loss (physical):",
                float(loss_physical),
            )

            plot_rollout_comparison(
                x_cord=x_cord.cpu().numpy().squeeze(),
                traj_pred=traj_pred[0].cpu().numpy(),
                traj_zero=traj_zero[:, 0, :].cpu().numpy(),
                u_t0=u0[0].cpu().numpy(),
                u_tf=u_tf[0].cpu().numpy(),
                control_pred=controls_pred[0].cpu().numpy(),
                step=max(1, nsteps // 10),
                save_path=PLOTS_DIR / f"part_7_rollout_comparison_sample_{idx}.png",
            )


if __name__ == "__main__":
    main()
