import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../stability_l4dc")
)

import torch
import numpy as np
from lpv import lpv_batched, lpv
from eigen_plots import compute_eigenvalues, plot_eigenvalues
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg as LA
import neuromancer.activations as nact
import slim
from neuromancer import blocks
import pandas as pd
from mk_plots import plot_eigenvalues_set, plot_phase_portrait_hidim, plot_singular_values, plot_Jacobian_norms


if __name__ == "__main__":

    SEED = 410
    nx = 64

    linmap = slim.linear.maps["damp_skew_symmetric"]
    # nonlin = torch.nn.ReLU
    nonlin = torch.nn.Identity
    fx = blocks.MLP(
        nx,
        nx,
        bias=False,
        linear_map=linmap,
        nonlin=nonlin,
        hsizes=[nx] * 4,
        linargs=dict(sigma_min=0.5, sigma_max=1.0, real=False),
    )
    plot_singular_values(fx, nx)
    plot_eigenvalues_set(fx, nx)
    # plot_phase_portrait_hidim(fx, nx, limits=(-6, 6))
    _, _, _, _ = plot_Jacobian_norms(fx, nx, limits=(-6, 6))
    plt.show()

    # test weight
    Astars_np_1 = fx.linear[0].effective_W().detach().numpy()
    Astars_np_2 = fx.linear[1].effective_W().detach().numpy()
    # Astars_np = np.dot(Astars_np_1, Astars_np_2)
    Astars_np = np.matmul(Astars_np_1, Astars_np_2)

    eigvals = compute_eigenvalues([Astars_np_1])
    plot_eigenvalues(eigvals)
    fig, ax = plt.subplots(1, 1)
    im1 = ax.imshow(Astars_np_1, vmin=abs(Astars_np_1).min(), vmax=abs(Astars_np_1).max(), cmap=plt.cm.CMRmap)
    ax.set_title('mean($A^{\star}}$)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    eigvals = compute_eigenvalues([Astars_np_2])
    plot_eigenvalues(eigvals)
    fig, ax = plt.subplots(1, 1)
    im1 = ax.imshow(Astars_np_2, vmin=abs(Astars_np_2).min(), vmax=abs(Astars_np_2).max(), cmap=plt.cm.CMRmap)
    ax.set_title('mean($A^{\star}}$)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    eigvals = compute_eigenvalues([Astars_np])
    plot_eigenvalues(eigvals)
    fig, ax = plt.subplots(1, 1)
    im1 = ax.imshow(Astars_np, vmin=abs(Astars_np).min(), vmax=abs(Astars_np).max(), cmap=plt.cm.CMRmap)
    ax.set_title('mean($A^{\star}}$)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    linmaps = ['spectral', "damp_skew_symmetric", "skew_symetric", "symplectic",
               'linear', "gershgorin", "pf", "softSVD"]

    activations = {
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "identity": torch.nn.Identity,
    }

    limits = (-6, 6)
    samples = 100

    for linmap in linmaps:
        for act_name, act in activations.items():
            fx = blocks.MLP(
                nx,
                nx,
                bias=False,
                linear_map=slim.linear.maps[linmap],
                nonlin=act,
                hsizes=[nx] * 4,
                linargs=dict(sigma_min=0.5, sigma_max=1.0, real=False),
            )
            plot_singular_values(fx, nx)
            plot_eigenvalues_set(fx, nx)
            # plot_phase_portrait_hidim(fx, nx, limits=(-6, 6))
            _, _, _, _ = plot_Jacobian_norms(fx, nx, limits=(-6, 6))
            plt.show()

            Y = []
            Y_astar = []
            Errors = []
            for k in range(samples):
                x = torch.rand(1, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
                Y.append(fx(x).squeeze())
                Astar, Astar_b, bstar, *_ = lpv(fx, x)
                Y_astar.append(torch.matmul(x, Astar_b) + bstar)
            y_astar = torch.stack(Y_astar)
            y_fx = torch.stack(Y)
            lin_error = torch.sum(y_fx - y_astar)
            Errors.append(lin_error)
            print(f"linearization error for {linmap} {act_name} is:{lin_error}")