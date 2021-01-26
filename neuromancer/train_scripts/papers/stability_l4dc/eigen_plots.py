import os
import warnings

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, writers
import slim
import torch
from torch import nn
import dill

from neuromancer import blocks
from neuromancer.datasets import EmulatorDataset

os.sys.path.append("neuromancer/train_scripts")
os.sys.path.append("neuromancer/train_scripts/lpv_l4dc")
from lpv import lpv

DENSITY_PALETTE = sns.color_palette("crest_r", as_cmap=True)
DENSITY_FACECLR = DENSITY_PALETTE(0.01)

sns.set_theme(style="white")


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
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(
            x=eigvals.real,
            y=eigvals.imag,
            fill=True,
            levels=50,
            thresh=0,
            cmap=DENSITY_PALETTE,
            ax=ax,
        )
    """
    sns.scatterplot(
        x=eigvals.real, y=eigvals.imag, alpha=0.5, ax=ax, color="white", s=7
    )
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        plt.close()

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
    items = zip(mat_list, eigval_list)
    fig, ax = plt.subplots(1, 2)
    animator = FuncAnimation(
        fig,
        plot_matrix_and_eigvals,
        items,
        fargs=(ax,),
        interval=16.66666,
        repeat_delay=3000,
        blit=False,
    )
    writer = writers["ffmpeg"](
        fps=60, bitrate=512000, codec="h264_mf", extra_args=["-s", "1280x960"]
    )
    if fname is not None:
        animator.save(fname, writer=writer, dpi=200)
        plt.close()


def gen_model_visuals(
    model,
    system,
    eigval_plot_fname="eigvals.svg",
    anim_fname="state_trajectory.mp4",
):
    fx = model.components[1].fx
    fu = model.components[1].fu
    estim = model.components[0]

    data = EmulatorDataset(system, nsim=1200, seed=50)
    loop_data = data.dev_loop

    A_stars = []
    eigvals = []
    x = estim(loop_data)["x0_estim"]
    for u in loop_data["Up"]:
        _, A_star_b, _, _, _, _, _ = lpv(fx, x)
        x = fx(x) + fu(u)  # torch.matmul(x, A_star) + fu(u)
        A_stars += [A_star_b.detach().cpu().numpy()]

    eigvals = compute_eigenvalues(A_stars)
    plot_eigenvalues(eigvals, fname=eigval_plot_fname)
    plot_matrix_eigval_anim(A_stars, eigvals, fname=anim_fname)


def despine_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


if __name__ == "__main__":
    print("Testing visualizations on random matrices...")
    random_matrices = [np.random.random(size=(3, 3)) / 2.0 for _ in range(100)]
    eigvals = compute_eigenvalues(random_matrices)

    plot_eigenvalues(eigvals, fname="random_matrices.svg")
    plot_matrix_eigval_anim(random_matrices, eigvals, fname="random_matrices.mp4")