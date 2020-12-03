"""
Utilities for post-hoc analysis of neural networks.
"""

import os
import warnings

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, writers
import torch

REGION_PALETTE = sns.color_palette("light:#00f", as_cmap=True)
DENSITY_PALETTE = sns.color_palette("crest_r", as_cmap=True)
DENSITY_FACECLR = DENSITY_PALETTE(0.01)

sns.set_theme(style="white")


def despine_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def compute_eigenvalues(matrices):
    eigvals = []
    for m in matrices:
        assert len(m.shape) == 2
        if not m.shape[0] == m.shape[1]:
            s = np.linalg.svd(m.T, compute_uv=False)
            lmbda = np.sqrt(s)
        else:
            lmbda, _ = np.linalg.eig(m.T)
        eigvals += [lmbda]
    return eigvals


def plot_eigenvalues(eigvals, ax=None, fname=None):
    """
    Plot eigenvalue spectra given list or ndarray of matrix eigenvalues.
    """
    if type(eigvals) == list:
        eigvals = np.concatenate(eigvals)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.clear()
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_aspect(1)
    ax.set_facecolor(DENSITY_FACECLR)
    patch = mpatches.Circle(
        (0, 0),
        radius=1,
        alpha=0.6,
        fill=False,
        ec=(0, 0.7, 1, 1),
        lw=2,
    )
    ax.add_patch(patch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(
            x=eigvals.real,
            y=eigvals.imag,
            fill=True,
            levels=100,
            thresh=0,
            cmap=DENSITY_PALETTE,
            ax=ax,
        )
    sns.scatterplot(
        x=eigvals.real, y=eigvals.imag, alpha=0.5, ax=ax, color="white", s=7
    )
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    return [ax]


def plot_matrix_and_eigvals(item, ax=None):
    matrix, eigvals = item
    if ax is None:
        _, ax = plt.subplots(1, 2)
    for a in ax:
        a.clear()

    ax[0].imshow(matrix, cmap="viridis")
    ax[1] = plot_eigenvalues(eigvals, ax[1])[0]

    despine_axis(ax[0])
    despine_axis(ax[1])

    return [*ax]


def plot_matrix_eigval_anim(mat_list, eigval_list, fname=None):
    """
    Given a list of matrices and their eigenvalues, plots an animation of the
    matrices and their corresponding eigenvalue spectra.
    """
    items = zip(mat_list, eigval_list)
    fig, ax = plt.subplots(1, 2)
    animator = FuncAnimation(
        fig,
        plot_matrix_and_eigvals,
        items,
        fargs=(ax,),
        interval=16.666,
        repeat_delay=3000,
        blit=False,
    )
    writer = writers["ffmpeg"](
        fps=60, bitrate=512000, codec="h264_mf", extra_args=["-s", "1280x960"]
    )
    if fname is not None:
        animator.save(fname, writer=writer, dpi=200)
        plt.close()


def plot_model_phase_portrait(
    model,
    x_lims=(0, 1),
    y_lims=(0, 1),
    step=0.05,
    data=None,
    initial_states=[],
    t=100,
    regions=True,
    fname=None,
):
    """
    Generate phase portrait using a model function.

    Accepts a function will be evaluated for one step using a grid of initial
    conditions determined by `x_lims`, `y_lims`, and `step`.

    If `initial_states` is specified, trajectories over `t` steps will
    be plotted over portrait.
    """
    _, ax = plt.subplots()
    if regions:
        X, Y = torch.meshgrid(
            torch.arange(*x_lims, (x_lims[1] - x_lims[0]) / 1024),
            torch.arange(*y_lims, (y_lims[1] - y_lims[0]) / 1024),
        )
        grid = torch.stack((X.flatten(), Y.flatten())).T
        grid = model(grid)

        U, V = grid.T.reshape(2, *X.shape).detach().cpu().numpy()

        U = -X + U
        V = -Y + V

        magnitudes = np.stack((U, V), axis=-1)
        magnitudes = np.linalg.norm(magnitudes, ord=2, axis=-1).T[::-1, :]
        ax.imshow(
            magnitudes,
            extent=[
                x_lims[0] - step / 2,
                x_lims[1] - step / 2,
                y_lims[0] - step / 2,
                y_lims[1] - step / 2,
            ],
            interpolation="bicubic",
            cmap=REGION_PALETTE,
        )

    X, Y = torch.meshgrid(torch.arange(*x_lims, step), torch.arange(*y_lims, step))
    grid = torch.stack((X.flatten(), Y.flatten())).T
    grid = model(grid)

    U, V = grid.T.reshape(2, *X.shape).detach().cpu().numpy()

    U = -X + U
    V = -Y + V

    # plot vector field
    ax.quiver(
        X, Y, U, V, angles="uv", pivot="mid", width=0.002, headwidth=4, headlength=5
    )

    # plot data over phase portrait
    if data is not None:
        data_states = data["Yp"].squeeze(1).cpu().numpy()
        ax.scatter(data_states[:, 0], data_states[:, 1], s=2, alpha=0.7, c="red")

    # plot state trajectories over phase space if initial states given
    if len(initial_states) > 0:
        states = torch.empty(t + 1, *initial_states.shape, requires_grad=False)
        states[0, :, :] = initial_states
        for i in range(t):
            states[i + 1, :, :] = model(states[i, :, :])
        states = states.transpose(1, 2).detach().cpu().numpy()
        ax.plot(states[:, 0], states[:, 1], marker="o", ms=3)

    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))

    if fname is not None:
        plt.savefig(fname)
        plt.close()

    return [ax]