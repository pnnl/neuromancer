import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../stability_l4dc")
)

import torch
import numpy as np
from neuromancer.analysis import compute_eigenvalues, plot_matrix_and_eigvals
from lpv import lpv_batched
import matplotlib.pyplot as plt

def plot_phase_portrait_hidim(fx, nx, limits=(-6, 6), nsamples=1000, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)

    XY = torch.empty(nsamples, 2)
    UV = torch.empty(nsamples, 2)
    for i, (Astar, x_sample) in enumerate(zip(Astars, x)):
        z_sample = torch.matmul(Astar, x_sample)
        u, s, vh = torch.svd(Astar)
        u = u[:2, ...]
        XY[i, ...] = torch.matmul(u, x_sample)
        UV[i, ...] = torch.matmul(u, z_sample)

    X, Y = XY.T.detach().numpy()
    U, V = UV.T.detach().numpy()

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(
        X, Y, U, V, angles="uv", pivot="mid", width=0.002, headwidth=4, headlength=5
    )

    ax.set_aspect(1)

    return [ax]


def plot_singular_values(fx, nx, limits=(-6, 6), nsamples=10, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)

    fig, ax = plt.subplots()
    for i, Astar in enumerate(Astars):
        _, s, _ = torch.svd(Astar, compute_uv=False)
        plt.plot(np.arange(1, nx + 1, 1), s.T.detach().numpy())
    ax.set_yscale("log")

    return [ax]


if __name__ == "__main__":
    # TODO(lltt): plot singular values
    import slim
    from neuromancer import blocks

    nx = 64

    linmap = slim.linear.maps["gershgorin"]
    fx = blocks.MLP(
        nx,
        nx,
        bias=False,
        linear_map=linmap,
        nonlin=torch.nn.SELU,
        hsizes=[nx] * 4,
        linargs=dict(sigma_min=0.9, sigma_max=1.1, real=False),
    )


    plot_singular_values(fx, nx)
    plt.show()
    plot_phase_portrait_hidim(fx, nx, limits=(-6, 6))