"""
Continuous Unscented Transform — Example: Van der Pol Oscillator
================================================================

Demonstrates three use cases for uncertainty propagation through
stochastic dynamics using the ContinuousUT API, plus a fourth
workflow ("Use Case 2b") that plugs a trained NeuroMANCER neural ODE
into the framework.

Use Cases
---------
1. **Propagate IC uncertainty** — known drift, zero diffusion, no training.
2. **Learn diffusion** — known drift, learn g2 from Monte-Carlo data.
3. **Full system identification** — learn both g1 (drift) and g2 (diffusion).
2b. **Neural ODE as drift** — train a neural ODE, then use it as g1.

System
------
Van der Pol oscillator with additive noise:

    dx1 = x2 dt + sigma dW1
    dx2 = [mu (1 - x1^2) x2 - x1] dt + sigma dW2

Run
---
    python continuous_ut_example.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# NeuroMANCER
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks as nm_blocks
from neuromancer.dataset import DictDataset

# ContinuousUT module
from neuromancer.dynamics.continuous_ut import (
    ContinuousUT,
    TrainingConfig,
    PropagationResult,
    PreciseLogger,
    simulate_sde,
    generate_moment_data,
)

torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# Ground truth: Van der Pol drift
# =====================================================================

class VanDerPolDrift(nn.Module):
    """Van der Pol oscillator drift."""

    def __init__(self, mu_param: float = 1.0):
        super().__init__()
        self.mu_param = mu_param

    def forward(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.stack(
            [x2, self.mu_param * (1 - x1**2) * x2 - x1], dim=-1
        )


vdp = VanDerPolDrift(mu_param=1.0)
nx = 2
dt = 0.01
mu0 = [2.0, 0.0]
var0 = [0.01, 0.01]
sigma_true = 0.1


# =====================================================================
# Use Case 1: Propagate IC Uncertainty Through Known Dynamics
# =====================================================================

print("\n" + "=" * 70)
print("USE CASE 1: Propagate IC Uncertainty (deterministic, no training)")
print("=" * 70)

model1 = ContinuousUT.from_known_dynamics(vdp, nx=nx, dt=dt)
model1.summary()

result1 = model1.propagate(mu0, var0, n_steps=500)
mc1 = simulate_sde(vdp, [0.0, 0.0], mu0, var0, 5.0, dt,
                    n_particles=10_000, seed=42)

print(f"Final mean (UT):  {result1.mu[-1]}")
print(f"Final mean (MC):  {mc1.mu[-1]}")
print(f"Final var  (UT):  {result1.var[-1]}")
print(f"Final var  (MC):  {mc1.var[-1]}")

model1.plot(result1, reference=mc1,
            title="Use Case 1: IC Uncertainty Propagation (Deterministic)",
            dim_names=["$x_1$", "$x_2$"])


# =====================================================================
# Use Case 2: Known Drift, Learn Diffusion
# =====================================================================

print("\n" + "=" * 70)
print("USE CASE 2: Known Drift, Learn Diffusion")
print("=" * 70)

print("Generating moment-transition data...")
train_loader2, dev_loader2, dev_data2, (xn2, xn_next2) = generate_moment_data(
    drift_fn=vdp, sigma=sigma_true, nx=nx, base_x0=np.array(mu0),
    n_ensembles=256, n_particles=256, t_per_ensemble=2.0,
    dt=dt, batch_size=64, seed=42,
)

model2 = ContinuousUT.from_known_drift(
    drift_fn=vdp, nx=nx, dt=dt,
    train_cfg=TrainingConfig(
        g1_warmup_epochs=0,
        joint_epochs=100,
        patience=20,
    ),
)
model2.summary()
model2.fit(train_loader2, dev_loader2, dev_data2,
           sigma_from_data=(xn2, xn_next2))

result2 = model2.propagate(mu0, var0, n_steps=500)
mc2 = simulate_sde(vdp, [sigma_true] * nx, mu0, var0, 5.0, dt,
                    n_particles=10_000, seed=42)

mse_mu = np.mean((result2.mu - mc2.mu) ** 2)
mse_var = np.mean((result2.var - mc2.var) ** 2)
print(f"Mean MSE:     {mse_mu:.6f}")
print(f"Variance MSE: {mse_var:.6f}")

model2.plot(result2, reference=mc2,
            title="Use Case 2: Learned Diffusion (Known Drift)")


# =====================================================================
# Use Case 3: Full System Identification
# =====================================================================

print("\n" + "=" * 70)
print("USE CASE 3: Full System Identification (learn g1 + g2)")
print("=" * 70)

print("Generating moment-transition data...")
train_loader3, dev_loader3, dev_data3, (xn3, xn_next3) = generate_moment_data(
    drift_fn=vdp, sigma=sigma_true, nx=nx, base_x0=np.array(mu0),
    n_ensembles=512, n_particles=512, t_per_ensemble=2.0,
    dt=dt, batch_size=64, seed=42,
)

model3 = ContinuousUT.from_data(
    nx=nx, dt=dt,
    train_cfg=TrainingConfig(
        g1_warmup_epochs=40,
        joint_epochs=110,
        patience=30,
    ),
)
model3.summary()
model3.fit(train_loader3, dev_loader3, dev_data3,
           sigma_from_data=(xn3, xn_next3))

result3 = model3.propagate(mu0, var0, n_steps=500)
mc3 = simulate_sde(vdp, [sigma_true] * nx, mu0, var0, 5.0, dt,
                    n_particles=10_000, seed=42)

mse_mu = np.mean((result3.mu - mc3.mu) ** 2)
mse_var = np.mean((result3.var - mc3.var) ** 2)
print(f"Mean MSE:     {mse_mu:.6f}")
print(f"Variance MSE: {mse_var:.6f}")

model3.plot(result3, reference=mc3,
            title="Use Case 3: Full System ID (Learned g1 + g2)")


# =====================================================================
# Side-by-side comparison
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (res, mc_ref, lbl) in zip(axes, [
    (result1, mc1, "UC1: IC Propagation"),
    (result2, mc2, "UC2: Learn Diffusion"),
    (result3, mc3, "UC3: Full System ID"),
]):
    ax.plot(mc_ref.time, mc_ref.mu[:, 0], "k-", lw=1.5, label="MC truth")
    ax.plot(res.time, res.mu[:, 0], "b--", lw=1.5, label="UT model")
    lo, hi = res.confidence_band(2)
    ax.fill_between(res.time, lo[:, 0], hi[:, 0], alpha=0.2, color="blue")
    ax.set_title(lbl)
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle("$x_1$ Mean: All Use Cases", fontweight="bold")
plt.tight_layout()
plt.show()


# =====================================================================
# Use Case 2b: Neural ODE as g1, then learn diffusion
# =====================================================================

print("\n" + "=" * 70)
print("USE CASE 2b: Train Neural ODE (g1) --> ContinuousUT (learn g2)")
print("=" * 70)

# --- Step 1: Generate stochastic trajectories ---

gt_drift = VanDerPolDrift(mu_param=1.0)
ts = dt


def simulate_stochastic_trajectories(
    drift_fn, sigma, nx, n_traj, n_steps, dt,
    x0_mean=None, x0_std=0.5, seed=42,
):
    """Euler–Maruyama simulation of raw stochastic trajectories."""
    np.random.seed(seed)
    if x0_mean is None:
        x0_mean = np.zeros(nx)
    trajectories = np.zeros((n_traj, n_steps + 1, nx))
    sigmas = np.array([sigma] * nx)
    for i in range(n_traj):
        x = x0_mean + x0_std * np.random.randn(nx)
        trajectories[i, 0] = x
        with torch.no_grad():
            for t in range(n_steps):
                drift = drift_fn(
                    torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                )
                drift = drift.squeeze(0).numpy()
                dW = np.random.randn(nx) * np.sqrt(dt)
                x = x + drift * dt + sigmas * dW
                trajectories[i, t + 1] = x
    return trajectories


n_traj = 200
n_steps_per_traj = 200  # 2 seconds at dt=0.01

print("Generating stochastic trajectories...")
trajs = simulate_stochastic_trajectories(
    gt_drift, sigma_true, nx, n_traj, n_steps_per_traj, ts,
    x0_mean=np.array([2.0, 0.0]), x0_std=0.5,
)
print(f"Trajectories shape: {trajs.shape}")

# Quick plot of trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
time_axis = np.arange(n_steps_per_traj + 1) * ts
for i in range(min(20, n_traj)):
    axes[0].plot(time_axis, trajs[i, :, 0], "gray", alpha=0.3, lw=0.5)
    axes[1].plot(time_axis, trajs[i, :, 1], "gray", alpha=0.3, lw=0.5)
axes[0].set_title("$x_1$ trajectories")
axes[0].set_xlabel("Time")
axes[1].set_title("$x_2$ trajectories")
axes[1].set_xlabel("Time")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.suptitle(
    f"Raw stochastic data ({n_traj} trajectories, $\\sigma$={sigma_true})",
    fontweight="bold",
)
plt.tight_layout()
plt.show()


# --- Step 2: Train a Neural ODE ---

nsteps_node = 2


def trajectories_to_neuromancer_data(trajs, nsteps, batch_size=100):
    """Chop trajectories into windows for neuromancer training."""
    n_traj, T, nx_dim = trajs.shape
    windows = []
    for i in range(n_traj):
        n_windows = (T - 1) // nsteps
        for j in range(n_windows):
            start = j * nsteps
            windows.append(trajs[i, start : start + nsteps + 1])
    windows = np.array(windows)
    print(f"  Created {len(windows)} windows of length {nsteps + 1}")

    idx = np.random.permutation(len(windows))
    n_train = int(0.8 * len(idx))
    train_w = torch.tensor(windows[idx[:n_train]], dtype=torch.float32)
    dev_w = torch.tensor(windows[idx[n_train:]], dtype=torch.float32)

    train_data = DictDataset(
        {"X": train_w, "xn": train_w[:, 0:1, :]}, name="train"
    )
    dev_data = DictDataset(
        {"X": dev_w, "xn": dev_w[:, 0:1, :]}, name="dev"
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        collate_fn=train_data.collate_fn, shuffle=True,
    )
    dev_loader = DataLoader(
        dev_data, batch_size=batch_size,
        collate_fn=dev_data.collate_fn, shuffle=False,
    )
    test_traj = torch.tensor(trajs[0:1], dtype=torch.float32)
    test_data = {"X": test_traj, "xn": test_traj[:, 0:1, :]}
    return train_loader, dev_loader, test_data


print("Preparing neural ODE training data...")
node_train, node_dev, node_test = trajectories_to_neuromancer_data(
    trajs, nsteps_node, batch_size=100
)

# Define and train the neural ODE
fx = nm_blocks.MLP(
    nx, nx, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.SiLU,
    hsizes=[64, 64, 64],
)

fxRK4 = integrators.RK4(fx, h=ts)
node_model = Node(fxRK4, ["xn"], ["xn"], name="NeuralODE")
node_system = System([node_model], name="node_system", nsteps=nsteps_node)

x_true = variable("X")
x_pred = variable("xn")[:, :-1, :]
xFD = x_true[:, 1:, :] - x_true[:, :-1, :]
xhatFD = x_pred[:, 1:, :] - x_pred[:, :-1, :]

ref_loss = (x_pred == x_true) ** 2
ref_loss.name = "ref_loss"
fd_loss = 2.0 * (xFD == xhatFD) ** 2
fd_loss.name = "fd_loss"

node_loss = PenaltyLoss([ref_loss, fd_loss], [])
node_problem = Problem([node_system], node_loss)

print(f"Neural ODE parameters: {sum(p.numel() for p in fx.parameters()):,}")
print("Training neural ODE on stochastic trajectory data...")

node_optimizer = torch.optim.Adam(node_problem.parameters(), lr=1e-3)
node_logger = PreciseLogger(
    args=None, savedir="./logs_node", verbosity=10,
    stdout=["dev_loss", "train_loss"], precision=6,
)
node_trainer = Trainer(
    node_problem, node_train, node_dev, node_test, node_optimizer,
    patience=30, warmup=20, epochs=200,
    eval_metric="dev_loss", train_metric="train_loss",
    dev_metric="dev_loss", test_metric="dev_loss",
    logger=node_logger,
)
best_node = node_trainer.train()
node_problem.load_state_dict(best_node)
print("Neural ODE training complete.")

# Evaluate neural ODE vs ground truth
fx.eval()
gt_drift.eval()
eval_points = torch.tensor([
    [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0],
    [1.0, 1.0], [-1.0, -1.0], [1.5, -0.5], [-0.5, 1.5],
], dtype=torch.float32)

with torch.no_grad():
    learned_drift = fx(eval_points)
    true_drift = gt_drift(eval_points)

print("\nDrift comparison at sample points:")
print(f"  {'Point':>20s}  {'True drift':>25s}  {'Learned drift':>25s}  {'Error':>10s}")
for i in range(len(eval_points)):
    pt = eval_points[i].numpy()
    td = true_drift[i].numpy()
    ld = learned_drift[i].numpy()
    err = np.linalg.norm(td - ld)
    print(f"  {str(pt):>20s}  {str(np.round(td, 3)):>25s}  "
          f"{str(np.round(ld, 3)):>25s}  {err:>10.4f}")


# --- Step 3: Plug neural ODE into ContinuousUT ---

# First, IC uncertainty propagation (Use Case 1 with neural ODE)
model_node_uc1 = ContinuousUT.from_known_dynamics(fx, nx=nx, dt=ts)
model_node_uc1.summary()

result_node_uc1 = model_node_uc1.propagate([2.0, 0.0], [0.01, 0.01],
                                             n_steps=500)
mc_node_uc1 = simulate_sde(
    gt_drift, [0.0, 0.0], [2.0, 0.0], [0.01, 0.01],
    5.0, ts, n_particles=10_000, seed=42,
)

print(f"\nNeural ODE IC propagation:")
print(f"  Final mean (UT):  {result_node_uc1.mu[-1]}")
print(f"  Final mean (MC):  {mc_node_uc1.mu[-1]}")

# Now learn diffusion with the neural ODE as g1
print("\nGenerating moment-transition data for g2 training...")
train_loader_g2, dev_loader_g2, dev_data_g2, (xn_all, xn_next_all) = \
    generate_moment_data(
        drift_fn=gt_drift, sigma=sigma_true, nx=nx,
        base_x0=np.array([2.0, 0.0]),
        n_ensembles=256, n_particles=256,
        t_per_ensemble=2.0, dt=ts, batch_size=64, seed=42,
    )

model_node = ContinuousUT.from_known_drift(
    drift_fn=fx, nx=nx, dt=ts,
    train_cfg=TrainingConfig(
        g1_warmup_epochs=0,
        joint_epochs=100,
        patience=20,
        verbosity=5,
    ),
)
model_node.summary()

print("\nTraining g2 with neural ODE as frozen g1...")
model_node.fit(
    train_loader_g2, dev_loader_g2, dev_data_g2,
    sigma_from_data=(xn_all, xn_next_all),
)


# --- Step 4: Evaluate ---

result_node = model_node.propagate([2.0, 0.0], [0.01, 0.01], n_steps=500)
mc_node = simulate_sde(
    gt_drift, [sigma_true] * nx, [2.0, 0.0], [0.01, 0.01],
    5.0, ts, n_particles=10_000, seed=42,
)

mse_mu = np.mean((result_node.mu - mc_node.mu) ** 2)
mse_var = np.mean((result_node.var - mc_node.var) ** 2)
print(f"\nNeural ODE g1 + Learned g2 vs MC ground truth:")
print(f"  Mean MSE:     {mse_mu:.6f}")
print(f"  Variance MSE: {mse_var:.6f}")

model_node.plot(
    result_node, reference=mc_node,
    title="Use Case 2b: Neural ODE g1 + Learned g2",
    dim_names=["$x_1$", "$x_2$"],
)

print("\nDone.")
