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
os.sys.path.append("neuromancer/train_scripts/L4DC_paper")
from autonomous_system import AutonomousSystem
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
    nx = 2
    square_maps = [
        ("gershgorin", -1.5, -1.1),
        ("gershgorin", 0.0, 1.0),
        ("gershgorin", 0.99, 1.1),
        ("gershgorin", 1.1, 1.5),
        # ("pf", 1.0, 1.0),
        # ("linear", 1.0, 1.0),
    ]

    activations = [
        ("relu", torch.nn.ReLU),
        ("selu", torch.nn.SELU),
        ("tanh", torch.nn.Tanh),
        ("sigmoid", torch.nn.Sigmoid),
    ]

    for nlayers in [1, 8]:
        for linmap, sigmin, sigmax in square_maps:
            for real in [True, False]:
                for bias in [True, False]:
                    for act_name, act in activations:
                        fx = blocks.MLP(
                            nx,
                            nx,
                            nonlin=act,
                            Linear=slim.linear.maps[linmap],
                            hsizes=[nx] * nlayers,
                            bias=bias,
                            linargs={
                                "sigma_min": sigmin,
                                "sigma_max": sigmax,
                                "real": real,
                            },
                        )

                        Astars = []
                        grid_x, grid_y = torch.meshgrid(
                            torch.arange(-6, 6, 0.5),
                            torch.arange(-6, 6, 0.5),
                        )
                        X = torch.stack((grid_x.flatten(), grid_y.flatten())).T
                        for x in X:
                            Astar, Astar_b, _, _, _, _, _ = lpv(fx, x)
                            Astars += [Astar_b.detach().cpu().numpy()]
                        eigvals = compute_eigenvalues(Astars)
                        plot_eigenvalues(eigvals, fname=f"spectrum_{linmap}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}{'_bias' if bias else ''}.png")
                        plot_matrix_eigval_anim(Astars, eigvals, fname=f"spectrum_{linmap}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}{'_bias' if bias else ''}.mp4")


    # NOTE: this prints a lot of stuff; use `lpv` for less printing
    # tut_system = AutonomousSystem(2, [2] * 4, nn.ReLU, linmap, 0.9, 1.0)

    fx = blocks.MLP(
        2,
        2,
        bias=False,
        Linear=linmap,
        nonlin=nn.Sigmoid,
        hsizes=[2] * 4,
        linargs=dict(sigma_min=0.9, sigma_max=1.0, real=True),
    )
    exit()

    # TODO: dead code follows
    PATH = "neuromancer/train_scripts/test/twotank_test.pth"
    SYSTEM = "TwoTank"
    PRNG = np.random.default_rng()

    model = torch.load(PATH, pickle_module=dill, map_location="cpu")
    fx = model.components[1].fx
    fu = model.components[1].fu

    print("Testing visualizations on random matrices...")
    random_matrices = [PRNG.random(size=(3, 3)) / 2.0 for _ in range(100)]
    eigvals = compute_eigenvalues(random_matrices)

    plot_eigenvalues(eigvals, fname="random_matrices.svg")
    plot_matrix_eigval_anim(random_matrices, eigvals, fname="random_matrices.mp4")

    print("Testing visualizations on model...")
    gen_model_visuals(
        model,
        SYSTEM,
        eigval_plot_fname="test_model_eigvals.svg",
        anim_fname="test_model.mp4",
    )
