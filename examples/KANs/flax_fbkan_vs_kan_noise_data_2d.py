import argparse
import os
import time
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import optax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTA = 1.9
COLORS = {"KAN": "#4e79a7", "FBKAN": "#59a14f"}

def bspline_basis(x, grid, k: int = 3):
    t = jnp.concatenate(
        [
            jnp.broadcast_to(grid[:, :1], (grid.shape[0], k)),
            grid,
            jnp.broadcast_to(grid[:, -1:], (grid.shape[0], k)),
        ],
        axis=1,
    )
    hi = grid[:, -1][:, None]
    lo = grid[:, 0][:, None]
    xe = jnp.clip(x[..., None], lo, hi - 1e-6 * (hi - lo))  # (..., d, 1)

    B = jnp.logical_and(xe >= t[:, :-1], xe < t[:, 1:]).astype(x.dtype)

    for deg in range(1, k + 1):
        n = t.shape[1] - deg - 1
        t_i = t[:, :n]
        t_i1 = t[:, 1 : 1 + n]
        t_id = t[:, deg : deg + n]          # t_{i+deg}
        t_id1 = t[:, deg + 1 : deg + 1 + n]  # t_{i+deg+1}
        den1 = t_id - t_i
        den2 = t_id1 - t_i1
        # Reuse the masks instead of recomputing abs/compare twice per branch.
        m1 = jnp.abs(den1) > 1e-8
        m2 = jnp.abs(den2) > 1e-8
        w1 = jnp.where(m1, (xe - t_i) / jnp.where(m1, den1, 1.0), 0.0)
        w2 = jnp.where(m2, (t_id1 - xe) / jnp.where(m2, den2, 1.0), 0.0)
        B = w1 * B[..., :n] + w2 * B[..., 1 : n + 1]
    return B


class KANLayer(nn.Module):
    out_features: int
    grid_size: int = 5
    spline_order: int = 3

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        G, k = self.grid_size, self.spline_order
        n_basis = G + k

        log_scale = self.param("log_grid_scale", nn.initializers.zeros, (d,))
        grid = jnp.linspace(-1.0, 1.0, G + 1)[None, :] * jnp.exp(log_scale)[:, None]
        x_clipped = jnp.clip(x, grid[:, 0], grid[:, -1])
        B = bspline_basis(x_clipped, grid, k=k)  # (..., d, n_basis)

        c = self.param(
            "spline_weight",
            nn.initializers.normal(stddev=0.1),
            (self.out_features, d, n_basis),
        )
        w_base = self.param(
            "base_weight", nn.initializers.lecun_normal(), (self.out_features, d)
        )

        y = jnp.einsum("...i,ji->...j", jax.nn.silu(x), w_base)
        y = y + jnp.einsum("...ib,jib->...j", B, c)
        return y


class KAN(nn.Module):
    features: Sequence[int]
    grid_size: int = 5
    spline_order: int = 3

    @nn.compact
    def __call__(self, x):
        for i, n_out in enumerate(self.features[1:]):
            x = KANLayer(
                n_out,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                name=f"kan_{i}",
            )(x)
        return x

def target_fn(xy):
    """f(x, y) = sin(4*pi*x^2) * sin(6*pi*y^2), xy (..., 2) in [0, 1]^2."""
    x, y = xy[..., 0], xy[..., 1]
    return jnp.sin(4.0 * jnp.pi * x**2) * jnp.sin(6.0 * jnp.pi * y**2)


def make_dataset(key, train_num=1000, test_num=1000, noise=0.05):
    """Uniform [0,1]^2 inputs; train labels get Gaussian noise of std `noise`.

    The notebook draws the same distribution with torch.rand / torch.randn.
    With a JAX PRNG the samples are statistically equivalent, not bit-identical.
    Device arrays are kept (no host sync before training).
    """
    k1, k2, k3 = jax.random.split(key, 3)
    x_train = jax.random.uniform(k1, (train_num, 2))
    x_test = jax.random.uniform(k2, (test_num, 2))
    y_train = target_fn(x_train)[:, None] + noise * jax.random.normal(
        k3, (train_num, 1)
    )
    y_test = target_fn(x_test)[:, None]
    return x_train, y_train, x_test, y_test


def window_functions(xy, num_domains, delta=DELTA, bounds=(0.0, 1.0)):
    """Partition-of-unity weights, port of neuromancer.window_functions (2D).

    n = sqrt(num_domains) windows per axis with centres mu_j evenly spaced in
    [xmin, xmax], sigma = (xmax - xmin) * (delta / 2) / (n - 1), and

        w_j(x) = ((1 + cos(pi * (x - mu_j) / sigma)) / 2)^2   if |z| < 1,
                 0                                            otherwise.

    Tensor product over the two axes, flattened to (batch, num_domains), then
    normalised so that sum_j w_j(x) = 1.

    The notebook derives xmin/xmax from the current batch's min/max. Here the
    domain is known to be [0,1]^2, so we fix the bounds: on the full training
    batch / mesh grid the values coincide, and train/eval weights no longer
    drift with finite-batch extrema.
    """
    n = int(np.sqrt(num_domains))
    if n * n != num_domains or n < 2:
        raise ValueError("num_domains must be a perfect square >= 4 for 2D inputs")
    lo, hi = bounds
    mu = jnp.linspace(lo, hi, n)[None, :]                     # (1, n)
    sigma = (hi - lo) * (delta / 2.0) / (n - 1) + 1e-12

    def w_1d(v):
        z = (v[:, None] - mu) / sigma                         # (batch, n)
        w = ((1.0 + jnp.cos(jnp.pi * z)) / 2.0) ** 2
        return jnp.where(jnp.abs(z) < 1.0, w, 0.0)

    w = jnp.einsum("bi,bj->bij", w_1d(xy[:, 0]), w_1d(xy[:, 1]))
    w = w.reshape(xy.shape[0], -1)                            # (batch, n*n)
    return w / (jnp.sum(w, axis=1, keepdims=True) + 1e-12)


class KANRegressor(nn.Module):
    """Single-domain KAN = KANBlock(num_domains=1) in the notebook."""

    features: Sequence[int]
    grid_size: int = 5
    spline_order: int = 3

    @nn.compact
    def __call__(self, xy):
        # [0,1]^2 -> [-1,1]^2, matching the spline grid range [-1, 1].
        z = 2.0 * xy - 1.0
        return KAN(
            features=self.features,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            name="kan",
        )(z)


class FBKANRegressor(nn.Module):
    """Finite-basis KAN, port of KANBlock(num_domains=4).

    Each subdomain owns an independent KAN; the outputs are blended with the
    partition of unity:  y(x) = sum_j w_j(x) * KAN_j(x).
    """

    features: Sequence[int]
    num_domains: int = 4
    grid_size: int = 5
    spline_order: int = 3
    delta: float = DELTA

    @nn.compact
    def __call__(self, xy):
        z = 2.0 * xy - 1.0
        w = window_functions(xy, self.num_domains, self.delta)  # (batch, D)
        outs = [
            KAN(
                features=self.features,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                name=f"domain_{j}",
            )(z)
            for j in range(self.num_domains)
        ]
        y = jnp.concatenate(outs, axis=-1)  # (batch, D)
        return jnp.sum(y * w, axis=-1, keepdims=True)

def train(model, params, x_train, y_train, x_test, y_test,
          steps, lr=0.02, cosine=False):
    """Compile once, then run the whole loop on device.

    lower().compile() separates compile time from stepping time; donate_argnums
    lets XLA reuse the state buffer; xs=None + length avoids materialising an
    arange for scan. Both train and dev MSE are recorded at every step.
    """
    schedule = (optax.cosine_decay_schedule(lr, steps, alpha=0.1) if cosine
                else optax.constant_schedule(lr))
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(schedule)
    )

    def train_loss(p):
        return jnp.mean((model.apply(p, x_train) - y_train) ** 2)

    def dev_loss(p):
        return jnp.mean((model.apply(p, x_test) - y_test) ** 2)

    loss_and_grad = jax.value_and_grad(train_loss)

    @partial(jax.jit, donate_argnums=(0,))
    def run(state):
        def one(s, _):
            loss, grads = loss_and_grad(s.params)
            return s.apply_gradients(grads=grads), (loss, dev_loss(s.params))

        return jax.lax.scan(one, state, None, length=steps)

    t0 = time.perf_counter()
    compiled = run.lower(state).compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    state, (losses, dev_losses) = compiled(state)
    jax.block_until_ready(losses)  # do not time an async launch only
    step_s = time.perf_counter() - t0

    return (state.params, np.asarray(losses), np.asarray(dev_losses),
            compile_s, step_s)


def test_metrics(model, params, x_test, y_test):
    """Normalized test MSE used in the notebook: mean((y-ŷ)^2)/mean(y^2)."""
    pred = model.apply(params, x_test)
    nmse = float(jnp.mean((pred - y_test) ** 2) / jnp.mean(y_test**2))
    return nmse, np.asarray(pred)


def make_eval_grid(model, n=100):
    """Jitted n x n surface evaluation, cached per model instance."""
    xs = jnp.linspace(0.0, 1.0, n)
    XX, YY = jnp.meshgrid(xs, xs)
    xy = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)
    exact = target_fn(xy).reshape(n, n)

    @jax.jit
    def predict(p):
        return model.apply(p, xy).reshape(n, n)

    def eval_fn(p):
        pred = predict(p)
        rel = float(jnp.linalg.norm(pred - exact) / jnp.linalg.norm(exact))
        return (np.asarray(XX), np.asarray(YY), np.asarray(pred),
                np.asarray(exact), rel)

    return eval_fn

def plot_training(results, y_test, path, steps):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    epochs = np.arange(steps)
    for name, r in results.items():
        c = COLORS.get(name)
        ax[0].plot(epochs, r["losses"], label=f"{name} train", color=c)
        ax[0].plot(epochs, r["dev_losses"], label=f"{name} dev", color=c,
                   ls="--", alpha=0.7)
        ax[1].scatter(y_test.ravel(), r["test_pred"].ravel(), label=name,
                      alpha=0.6, s=12, color=c)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE")
    ax[0].legend(fontsize=10)
    ax[0].grid(alpha=0.3)
    lo, hi = float(y_test.min()), float(y_test.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=2, label="Ideal fit")
    ax[1].set_xlabel("True values")
    ax[1].set_ylabel("Predictions")
    ax[1].legend(fontsize=10)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_surfaces(results, path):
    cmap = plt.get_cmap("viridis")
    first = next(iter(results.values()))
    XX, YY, EX = first["X"], first["Y"], first["exact"]
    names = list(results)

    def panel(ax, Z, title, vmin=None, vmax=None, contour=False):
        p = ax.pcolor(XX, YY, Z, vmin=vmin, vmax=vmax, cmap=cmap,
                      shading="auto")
        if contour:
            ax.contour(XX, YY, Z, colors="k", linewidths=0.5)
        fig.colorbar(p, ax=ax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    if len(names) == 2:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        panel(ax[0, 0], EX, "Exact", -1.0, 1.0, contour=True)
        panel(ax[0, 1], results[names[0]]["pred"], names[0], -1.0, 1.0,
              contour=True)
        panel(ax[0, 2], np.abs(results[names[0]]["pred"] - EX),
              f"{names[0]} absolute error")
        panel(ax[1, 1], results[names[1]]["pred"], names[1], -1.0, 1.0,
              contour=True)
        panel(ax[1, 2], np.abs(results[names[1]]["pred"] - EX),
              f"{names[1]} absolute error")
        mid = EX.shape[0] // 2
        ax[1, 0].plot(XX[0, :], EX[mid, :], "-k", label="Exact")
        for name in names:
            ax[1, 0].plot(XX[0, :], results[name]["pred"][mid, :],
                          label=name, alpha=0.8)
        ax[1, 0].set_xlabel("$x$ (centerline $y=0.5$)")
        ax[1, 0].set_ylabel("$f(x,y)$")
        ax[1, 0].set_title("Comparison along centerline")
        ax[1, 0].legend()
        ax[1, 0].grid(alpha=0.3)
    else:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        name = names[0]
        panel(ax[0, 0], EX, "Exact", -1.0, 1.0, contour=True)
        panel(ax[0, 1], results[name]["pred"], name, -1.0, 1.0, contour=True)
        panel(ax[1, 0], np.abs(results[name]["pred"] - EX),
              f"{name} absolute error")
        mid = EX.shape[0] // 2
        ax[1, 1].plot(XX[0, :], EX[mid, :], "-k", label="Exact")
        ax[1, 1].plot(XX[0, :], results[name]["pred"][mid, :], label=name)
        ax[1, 1].set_xlabel("$x$ (centerline $y=0.5$)")
        ax[1, 1].set_ylabel("$f(x,y)$")
        ax[1, 1].set_title("Comparison along centerline")
        ax[1, 1].legend()
        ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="KAN vs FBKAN on noisy 2D data (neuromancer p2 example)"
    )
    ap.add_argument("--model", choices=["both", "kan", "fbkan"], default="both")
    ap.add_argument("--steps", type=int, default=1001)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay instead of the notebook's constant lr")
    ap.add_argument("--width", type=int, default=5, help="Hidden layer width")
    ap.add_argument("--grid", type=int, default=5,
                    help="Number of B-spline grid intervals G")
    ap.add_argument("--order", type=int, default=3, help="B-spline order k")
    ap.add_argument("--domains", type=int, default=4,
                    help="FBKAN subdomains (perfect square for 2D)")
    ap.add_argument("--delta", type=float, default=DELTA,
                    help="Window overlap ratio")
    ap.add_argument("--train-num", type=int, default=1000)
    ap.add_argument("--test-num", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="Std of Gaussian noise on train labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--mesh", type=int, default=100, help="Evaluation grid size")
    ap.add_argument("--save-dir", default=".", help="Where to write the figures")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--cache-dir", default=None,
                    help="Persistent JAX compilation cache (reuse compiled kernels across runs)")
    args = ap.parse_args()

    if args.cache_dir:
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(args.cache_dir)  # must be set before any jit compilation

    os.makedirs(args.save_dir, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    k_data, k_kan, k_fbkan = jax.random.split(key, 3)
    x_train, y_train, x_test, y_test = make_dataset(
        k_data, args.train_num, args.test_num, args.noise
    )
    features = [2, args.width, 1]

    specs = []
    if args.model in ("both", "kan"):
        specs.append(("KAN", KANRegressor(
            features=features, grid_size=args.grid, spline_order=args.order), k_kan))
    if args.model in ("both", "fbkan"):
        specs.append(("FBKAN", FBKANRegressor(
            features=features, num_domains=args.domains, grid_size=args.grid,
            spline_order=args.order, delta=args.delta), k_fbkan))

    print(f"task: f(x,y)=sin(4*pi*x^2)*sin(6*pi*y^2) on [0,1]^2 | "
          f"train={args.train_num} (noise std={args.noise}) test={args.test_num}")
    print(f"KAN={features} G={args.grid} k={args.order} | Adam lr={args.lr} "
          f"({'cosine' if args.cosine else 'constant'}) steps={args.steps} | "
          f"FBKAN domains={args.domains} delta={args.delta}")

    results = {}
    for name, model, k in specs:
        params = model.init(k, jnp.zeros((1, 2)))
        n_param = sum(int(np.prod(v.shape)) for v in jax.tree.leaves(params))
        print(f"\n[{name}] params={n_param}")
        params, losses, dev_losses, compile_s, step_s = train(
            model, params, x_train, y_train, x_test, y_test,
            args.steps, lr=args.lr, cosine=args.cosine,
        )
        print(f"  JIT compile {compile_s:.2f}s | {args.steps} steps {step_s:.1f}s "
              f"({1e3 * step_s / args.steps:.1f} ms/step)   "
              f"loss: {losses[0]:.3e} -> {losses[-1]:.3e}")
        for ep in sorted(set(list(range(0, args.steps, args.log_every))
                             + [args.steps - 1])):
            print(f"  epoch {ep:4d}  train_loss {losses[ep]:.5f}  "
                  f"dev_loss {dev_losses[ep]:.5f}")
        nmse, pred = test_metrics(model, params, x_test, y_test)
        X, Y, P, E, rel = make_eval_grid(model, args.mesh)(params)
        print(f"  test MSE / mean(y_test^2) = {nmse:.6f}   "
              f"relative L2 (mesh) = {rel:.6f}")
        results[name] = dict(
            losses=losses, dev_losses=dev_losses, test_pred=pred,
            X=X, Y=Y, pred=P, exact=E, nmse=nmse, rel_l2=rel,
        )

    if not args.no_plot:
        p1 = os.path.join(args.save_dir, "kan_vs_fbkan_loss.png")
        p2 = os.path.join(args.save_dir, "kan_vs_fbkan_surface.png")
        plot_training(results, np.asarray(y_test), p1, args.steps)
        plot_surfaces(results, p2)
        print(f"\nFigures: {p1}\n         {p2}")


if __name__ == "__main__":
    main()
