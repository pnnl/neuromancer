# Continuous-Time Unscented Transform for Stochastic Dynamics

## Overview

This module propagates **mean and covariance** of uncertain states through
nonlinear stochastic ordinary differential equations (SDEs) of the form

$$
\mathrm{d}\mathbf{x} = g_1(\mathbf{x})\,\mathrm{d}t + \sqrt{2\,g_2(\mathbf{x})}\;\mathrm{d}\mathbf{W}
$$

where $g_1$ is the drift, $g_2$ is a state-dependent diffusion coefficient,
and $\mathbf{W}$ is a standard Wiener process.  In the learnable setting,
$g_1$ and/or $g_2$ are parameterised as neural networks
$g_{1,\theta_1}$ and $g_{2,\theta_2}$.

Rather than running expensive Monte Carlo ensembles, the **continuous-time
Unscented Transform (UT)** converts the stochastic propagation problem into a
deterministic ODE over the moments (mean $\mathbf{m}$ and covariance $\mathbf{P}$),
which can then be integrated with a standard solver (RK4).
The module also supports **learning** the drift and/or diffusion from data
using NeuroMANCER's training infrastructure.

---

## Mathematical background

### 1. The moment ODE

Given the SDE above, the time evolution of the first two moments is

$$
\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t} = \mathbb{E}[g_1(\mathbf{x})],
\qquad
\frac{\mathrm{d}\mathbf{P}}{\mathrm{d}t} = \mathrm{Cov}[\mathbf{x},\, g_1(\mathbf{x})] + \mathrm{Cov}[g_1(\mathbf{x}),\, \mathbf{x}] + 2\,\mathrm{diag}\!\big(g_2(\mathbf{m})\big).
$$

The expectations are intractable for nonlinear $g_1$. The standard
Unscented Transform approximates them with a deterministic set of
**sigma points**.

### 2. Sigma points (augmented state)

Because the noise enters the dynamics, the state is augmented with the noise
dimension: $\mathbf{z} = [\mathbf{x};\; \mathbf{w}]$ with $n_{\mathrm{aug}} = n_x + n_w$.
From the augmented mean $\mathbf{m}_a$ and covariance $\mathbf{P}_a$ we draw
$2 n_{\mathrm{aug}} + 1$ sigma points:

$$
\mathcal{Z}^{(0)} = \mathbf{m}_a, \qquad
\mathcal{Z}^{(i)} = \mathbf{m}_a + \gamma\, \mathbf{L}_{:,i}, \qquad
\mathcal{Z}^{(n_{\mathrm{aug}}+i)} = \mathbf{m}_a - \gamma\, \mathbf{L}_{:,i}
$$

where $\mathbf{L}$ is the Cholesky factor of $\mathbf{P}_a$ and
$\gamma = \sqrt{n_{\mathrm{aug}} + \lambda}$ with
$\lambda = \alpha^2(n_{\mathrm{aug}} + \kappa) - n_{\mathrm{aug}}$.

The augmented covariance is block-diagonal:

$$
\mathbf{P}_a = \begin{bmatrix} \mathbf{P} & 0 \\ 0 & \mathbf{I} \end{bmatrix}
$$

where the identity block represents the unit-variance Wiener increments.

### 3. Propagation through the augmented dynamics

Each sigma point is propagated through the augmented dynamics function:

$$
F_x^{(k)} = g_1\!\big(\mathcal{Z}_x^{(k)}\big) + \sqrt{2\,g_2\!\big(\mathcal{Z}_x^{(k)}\big)}\;\mathcal{Z}_w^{(k)}
$$

where $\mathcal{Z}_x$ and $\mathcal{Z}_w$ are the state and noise
partitions of the sigma point, respectively.

### 4. Recovering moment rates

The moment rates are recovered as UT-weighted statistics:

$$
\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t} = \sum_{k=0}^{2n_a} W_m^{(k)}\, F_x^{(k)}
$$

$$
\frac{\mathrm{d}\mathbf{P}}{\mathrm{d}t}
= \sum_{k=0}^{2n_a} W_c^{(k)} \Big[\big(\mathcal{Z}_x^{(k)} - \bar{\mathbf{x}}\big)\big(F_x^{(k)} - \bar{F}\big)^\top + \big(F_x^{(k)} - \bar{F}\big)\big(\mathcal{Z}_x^{(k)} - \bar{\mathbf{x}}\big)^\top\Big]
\;+\; 2\,\mathrm{diag}\!\big(g_2(\mathbf{m})\big)
$$

where $\bar{\mathbf{x}}$ and $\bar{F}$ are the weighted means of the sigma
points and their images, and $W_m^{(k)}, W_c^{(k)}$ are the standard UT weights.

### 5. Integration

The combined state $[\mathbf{m},\;\mathrm{vec}(\mathbf{P})]$ is integrated
forward in time using RK4.  At each RK4 stage the moment ODE above is
evaluated, making the full propagation a standard initial-value problem with
no stochastic sampling.

---

## What is learned

The drift and diffusion functions can each be either known (fixed) or
parameterised as neural networks:

$$
g_1(\mathbf{x}) \to g_{1,\theta_1}(\mathbf{x}), \qquad
g_2(\mathbf{x}) \to g_{2,\theta_2}(\mathbf{x})
$$

where $\theta_1$ and $\theta_2$ are the MLP weights for the drift and
diffusion networks respectively.  The table below shows which parameters
exist and which are trained in each use case:

| # | Use Case | $g_1$ | $g_2$ | Trainable params |
|---|----------|:-----:|:-----:|:----------------:|
| 1 | Propagate IC uncertainty | Known function | $0$ (zero) | $\emptyset$ (no training) |
| 2 | Learn diffusion | Known function (frozen) | $g_{2,\theta_2}$ (MLP + softplus) | $\theta_2$ only |
| 3 | Full system ID | $g_{1,\theta_1}$ (MLP) | $g_{2,\theta_2}$ (MLP + softplus) | $\theta_1, \theta_2$ |

A fourth workflow (**Use Case 2b**) first trains a NeuroMANCER neural ODE
$g_{1,\theta_1}$ on raw trajectory data, then freezes $\theta_1$ and plugs it
into Use Case 2 to learn $\theta_2$ — bridging the standard neural ODE
tutorial with uncertainty quantification.

---

## Use Case 1 in depth: reachability and feasible IC sets

Use Case 1 requires no training, but it is arguably the most directly
useful for **open-loop control design** and **safety analysis**.  The core
question it answers is:

> Given a known plant model $g_1$ and uncertainty in the initial condition
> $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \mathrm{diag}(\mathbf{v}_0))$,
> will the system stay within prescribed operating bounds
> $\mathbf{x}_{\min} \le \mathbf{x}(t) \le \mathbf{x}_{\max}$ for all
> $t \in [0, T]$?

The UT propagation gives you the time-evolving confidence envelope
$\boldsymbol{\mu}(t) \pm n_\sigma \sqrt{\mathbf{v}(t)}$ at negligible cost
compared to Monte Carlo.  This turns constraint checking into a simple
comparison:

$$
\boldsymbol{\mu}(t) - n_\sigma \sqrt{\mathbf{v}(t)} \;\ge\; \mathbf{x}_{\min}
\quad \text{and} \quad
\boldsymbol{\mu}(t) + n_\sigma \sqrt{\mathbf{v}(t)} \;\le\; \mathbf{x}_{\max}
\qquad \forall\; t \in [0, T].
$$

### Inverse design: permissible initial conditions

Because `propagate()` is fast and differentiable, you can also solve the
**inverse problem**: find the largest set of initial conditions
$(\boldsymbol{\mu}_0, \mathbf{v}_0)$ such that the propagated envelope
never violates the bounds.  Concretely, you can sweep over candidate
$\mathbf{v}_0$ values (or optimise them) and check whether the envelope
stays feasible — a form of **reachable-set characterization** without
expensive backward-reachability computation.

### Application: SIR epidemic models

This capability is particularly valuable in compartmental epidemic models
(SIR, SEIR, etc.), where:

- The **dynamics are known** (or well-calibrated), but the **initial
  conditions are uncertain** — the true number of infected individuals at
  the start of an outbreak is never known precisely.
- There are hard **capacity constraints** — e.g., the infected fraction
  $I(t)$ must stay below a hospital-capacity threshold $I_{\max}$.
- The question is inherently about the **envelope**, not the mean
  trajectory: a mean prediction of $I(t) < I_{\max}$ is meaningless if the
  $2\sigma$ upper band exceeds it.

With Use Case 1, a public-health planner can:

1. **Forward analysis.**  Given current surveillance uncertainty
   $\mathbf{v}_0$, propagate forward and check whether the
   $\pm 2\sigma$ band on $I(t)$ breaches $I_{\max}$.
2. **Inverse design.**  Ask "what is the maximum tolerable uncertainty
   in our initial infection estimate such that $I(t)$ stays below capacity
   with $2\sigma$ confidence?" — i.e., find the **permissible IC set**.
3. **Intervention timing.**  Compare envelopes under different
   intervention start times to find the latest point at which action still
   keeps the system in bounds.

In all cases, the UT propagation replaces a large Monte Carlo ensemble with
a single deterministic ODE integration — typically 100–1000× faster — making
interactive exploration and optimisation loops practical.

### Example: constraint checking

```python
model = ContinuousUT.from_known_dynamics(sir_drift, nx=3, dt=0.01)
result = model.propagate(mu0, var0, n_steps=1000)

# Check if 2-sigma upper band on I(t) stays below capacity
lo, hi = result.confidence_band(n_sigma=2)
I_max = 0.3  # hospital capacity as fraction
feasible = np.all(hi[:, 1] <= I_max)  # dim 1 = I compartment
print(f"Feasible: {feasible}")
```

---

## Training pipeline

Training (Use Cases 2 and 3) uses the standard NeuroMANCER
`Node → System → Problem → Trainer` stack.

### Data format

Training data consists of **moment transitions**: consecutive
$(\mathbf{m}_t, \mathbf{v}_t) \to (\mathbf{m}_{t+1}, \mathbf{v}_{t+1})$
pairs, where $\mathbf{v}$ is the diagonal of $\mathbf{P}$.  These are
obtained from Monte Carlo ensembles via `generate_moment_data()`.

### Sigma estimation for warm-up

Before training begins, the module needs an initial scalar estimate of the
diffusion strength $\hat{\sigma}$ to construct a `ConstantG2` block
($g_2 = \hat{\sigma}^2 / 2$) used during the drift warm-up phase.  This is
resolved through a three-level fallback:

1. **User-provided.**  Pass `sigma_init=0.15` to the factory constructor and
   it is used directly.
2. **Estimated from data.**  If `sigma_init` is `None` but
   `sigma_from_data=(xn_all, xn_next_all)` is passed to `fit()`, the module
   calls `estimate_sigma()`.  For each state dimension $d$, this computes the
   variance increment $\Delta v_d = v_{t+1,d} - v_{t,d}$ across all
   moment-transition pairs, takes the **median of the positive increments**,
   and estimates

   $$\hat{\sigma}_d = \sqrt{\mathrm{median}(\Delta v_d^{+}) \;/\; \Delta t}$$

   The intuition is that in the diffusion-dominated regime,
   $\mathrm{d}\mathrm{Var}/\mathrm{d}t \approx \sigma^2$, so the median
   positive increment divided by $\Delta t$ gives a robust estimate of
   $\sigma^2$, filtering out negative increments caused by the nonlinear
   drift contracting variance.  The final scalar estimate is the mean over
   dimensions: $\hat{\sigma} = \frac{1}{n_x}\sum_d \hat{\sigma}_d$.

3. **Default.**  If both are `None`, it falls back to $\hat{\sigma} = 0.1$.

The estimate only needs to be "good enough" — it sets the constant $g_2$
for Phase 1 warm-up, and the `LearnableG2` network $g_{2,\theta_2}$ replaces
it entirely in Phase 2.

### Two-phase schedule (Use Case 3)

Full system identification uses a curriculum to avoid the optimiser trying
to learn drift and diffusion simultaneously from a random initialisation:

1. **Phase 1 — Drift warm-up.**  Train $\theta_1$ (the drift MLP) while
   holding diffusion at the constant estimate: $g_2 = \hat{\sigma}^2/2$.
   The diffusion parameters $\theta_2$ are frozen
   (`requires_grad=False`).  This gives the drift network a reasonable
   initialisation before introducing the learnable diffusion.

2. **Phase 2 — Joint training.**  Unfreeze $\theta_2$ and optimise
   $\theta_1, \theta_2$ jointly.  A `SwitchableG2` block swaps from the
   constant warm-up value to the `LearnableG2` network $g_{2,\theta_2}$
   at the start of this phase.

For Use Case 2, Phase 1 is skipped (the drift is already known and frozen)
and only $\theta_2$ is optimised.

### Loss

The loss is a weighted sum of mean-tracking and variance-tracking MSE over
one-step moment predictions:

$$
\mathcal{L}(\theta) = w_\mu \|\hat{\mathbf{m}}_\theta - \mathbf{m}^*\|^2 + w_v \|\hat{\mathbf{v}}_\theta - \mathbf{v}^*\|^2
$$

where $\hat{\mathbf{m}}_\theta, \hat{\mathbf{v}}_\theta$ are the predicted
next-step moments from the UT integrator and $\mathbf{m}^*, \mathbf{v}^*$
are the empirical moments from Monte Carlo data.  $\theta$ denotes whichever
parameters are active: $\theta = \theta_2$ in Use Case 2, or
$\theta = \{\theta_1, \theta_2\}$ in Use Case 3.

---

## Architecture

```
src/neuromancer/dynamics/
└── continuous_ut.py        # All classes, utilities, and the ContinuousUT API

examples/SDEs/
└── continuous_ut_example.py  # Van der Pol demo (all four use cases)

test/
└── test_continuous_ut.py     # pytest unit tests
```

### Block subclasses

All blocks extend NeuroMANCER's `Block` base class:

| Block | Role |
|-------|------|
| `ContinuousUTDynamics` | The moment ODE right-hand side (sigma points → weighted statistics) |
| `ContinuousUTIntegrator` | Wrapper that converts between packed `(μ, v)` and flat `[m, vec(P)]` around a NeuroMANCER integrator |
| `DriftWrapper` | Adapts any `nn.Module` or callable as a `Block` for use as $g_1$ |
| `ZeroG2` | Zero diffusion (Use Case 1) |
| `ConstantG2` | Fixed $g_2 = \hat{\sigma}^2/2$ (warm-up phase; $\hat{\sigma}$ from estimation or user) |
| `LearnableG2` | $g_{2,\theta_2}$: MLP with softplus output ensuring positivity |
| `SwitchableG2` | Dispatches between `ConstantG2` and `LearnableG2` during the two-phase training schedule |

### Internal NeuroMANCER pipeline

During `fit()`, the module assembles:

```
ContinuousUTDynamics(g1, g2)  →  integrators.RK4  →  ContinuousUTIntegrator
        ↓
    Node(block, ["xn"], ["xn"])
        ↓
    System([node], nsteps=1)
        ↓
    Problem([system], PenaltyLoss)
        ↓
    Trainer(problem, ...)
```

During `propagate()`, a standalone `ContinuousUTIntegrator` is stepped
forward in a simple `torch.no_grad()` loop.

---

## Quick start

```python
from neuromancer.dynamics.continuous_ut import (
    ContinuousUT,
    TrainingConfig,
    simulate_sde,
    generate_moment_data,
)
```

### Use Case 1 — Propagate IC uncertainty (no training)

```python
model = ContinuousUT.from_known_dynamics(my_drift_fn, nx=2, dt=0.01)
result = model.propagate(mu0=[2.0, 0.0], var0=[0.01, 0.01], n_steps=500)
model.plot(result, title="IC Uncertainty")
```

### Use Case 2 — Learn diffusion from data

```python
train_loader, dev_loader, dev_data, (xn, xn_next) = generate_moment_data(
    drift_fn=my_drift_fn, sigma=0.1, nx=2,
    base_x0=np.array([2.0, 0.0]),
)

model = ContinuousUT.from_known_drift(
    drift_fn=my_drift_fn, nx=2, dt=0.01,
    train_cfg=TrainingConfig(joint_epochs=100, patience=20),
)
model.fit(train_loader, dev_loader, dev_data, sigma_from_data=(xn, xn_next))
result = model.propagate([2.0, 0.0], [0.01, 0.01], n_steps=500)
```

### Use Case 3 — Full system identification

```python
model = ContinuousUT.from_data(
    nx=2, dt=0.01,
    train_cfg=TrainingConfig(g1_warmup_epochs=40, joint_epochs=110),
)
model.fit(train_loader, dev_loader, dev_data, sigma_from_data=(xn, xn_next))
result = model.propagate([2.0, 0.0], [0.01, 0.01], n_steps=500)
```

### Use Case 2b — Neural ODE as drift

```python
# 1. Train a neuromancer neural ODE (fx) on trajectory data
# 2. Plug it in as frozen g1:
model = ContinuousUT.from_known_drift(drift_fn=fx, nx=2, dt=0.01)
model.fit(train_loader, dev_loader, dev_data, sigma_from_data=(xn, xn_next))
```

A complete working example (Van der Pol oscillator, all four workflows) is in
[`examples/SDEs/continuous_ut_example.py`](../../examples/SDEs/continuous_ut_example.py).

---

## Configuration reference

| Dataclass | Key fields | Default |
|-----------|-----------|---------|
| `TrainingConfig` | `lr`, `patience`, `warmup`, `g1_warmup_epochs`, `joint_epochs`, `mu_loss_weight`, `var_loss_weight` | `1e-3`, `30`, `10`, `40`, `110`, `1.0`, `0.1` |
| `UTConfig` | `alpha`, `beta`, `kappa` | `1.0`, `0.0`, `0.0` |
| `NetworkConfig` | `g1_hsizes`, `g2_hsizes`, `g2_min` | `[64,64,64]`, `[64,64]`, `1e-6` |

---

## References

1. S. J. Julier and J. K. Uhlmann, "Unscented filtering and nonlinear
   estimation," *Proceedings of the IEEE*, 2004.
2. J. O'Leary, J. A. Paulson, and A. Mesbah, "Stochastic physics-informed
   neural networks (SPINN): A moment-matching framework for learning hidden
   physics within stochastic differential equations," *arXiv preprint
   arXiv:2109.01621*, 2021.
3. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural
   networks," *Journal of Computational Physics*, 2019.
4. NeuroMANCER documentation: https://github.com/pnnl/neuromancer
