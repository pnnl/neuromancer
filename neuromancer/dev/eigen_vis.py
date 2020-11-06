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
from neuromancer.datasets import EmulatorDataset
os.sys.path.append("train_scripts")
from neuromancer.train_scripts.tutorial_system import LPV_net

DENSITY_PALETTE = sns.color_palette("crest_r", as_cmap=True)
DENSITY_FACECLR = DENSITY_PALETTE(
    0.01
)

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

    ax[0].imshow(matrix)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1] = plot_eigenvalues(eigvals, ax[1])[0]

    return [*ax]


def plot_Astar_anim(Astar_list, eigval_list, fname=None):
    items = zip(Astar_list, eigval_list)
    fig, ax = plt.subplots(1, 2)
    animator = FuncAnimation(
        fig,
        plot_matrix_and_eigvals,
        items,
        fargs=(ax,),
        interval=128,
        repeat_delay=3000,
        blit=False,
    )
    writer = writers["ffmpeg"](fps=20, bitrate=128000, codec="h264_mf")
    if fname is not None:
        animator.save(fname, writer=writer)


def plot_phase_portrait(
    fx,
    x_lims=(-3, 3),
    y_lims=(-3, 3),
    step=0.1,
    initial_states=[],
    t=100,
    fname=None,
):
    X, Y = np.meshgrid(np.arange(*x_lims, step), np.arange(*y_lims, step))

    g = np.stack((X.flatten(), Y.flatten()))
    g = torch.tensor(g, dtype=torch.float)
    g = fx(g.T)
    g = g.T.detach().cpu().numpy()
    U, V = g.reshape(2, *X.shape)

    # TODO(lltt): doesn't seem right to do this
    U = -X + U
    V = -Y + V

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, pivot="mid")

    # plot state trajectories over phase space if initial states given
    initial_states = torch.tensor(initial_states, dtype=torch.float)
    states = torch.empty(t+1, *initial_states.shape)
    states[0, :, :] = initial_states
    for i in range(t):
        states[i+1, :, :] = fx(states[i, :, :])
    states = states.transpose(1, 2).detach().cpu().numpy()
    ax.plot(states[:, 0], states[:, 1])
    ax.set_xlim((x_lims[0] - step/2, x_lims[1] - step/2))
    ax.set_ylim((y_lims[0] - step/2, y_lims[1] - step/2))
    if fname is not None:
        plt.savefig(fname)

    return [ax]


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
        A_star, _, _, _, w_net = LPV_net(fx, x, verbose=False)
        x = fx(x) + fu(u)  # torch.matmul(x, A_star) + fu(u)
        A_stars += [A_star.detach().cpu().numpy()]
        eigvals += [w_net]

    plot_eigenvalues(eigvals, fname=eigval_plot_fname)
    plot_Astar_anim(A_stars, eigvals, fname=anim_fname)


if __name__ == "__main__":
    PATH = "neuromancer/train_scripts/test/twotank_test.pth"
    SYSTEM = "TwoTank"
    PRNG = np.random.default_rng()

    model = torch.load(PATH, pickle_module=dill, map_location="cpu")
    fx = model.components[1].fx
    fu = model.components[1].fu

    # NOTE: will fail for models with nx > 2
    print("Testing model phase portrait...")
    initial_states = [
        #np.array([np.cos(x), np.sin(x)])*0.5 + np.array([0, 0])
        #for x in np.arange(0, 2 * np.pi, np.pi / 2)
        np.array([-2, 0]),
        np.array([0.5, -1])
    ]
    # spiral_A = torch.tensor([[0, 0.8], [-0.05, -.0135]], dtype=torch.float).T
    plot_phase_portrait(
        fx, # lambda x: torch.matmul(x, spiral_A),
        x_lims=(-2, 2.1),
        y_lims=(-2, 2.1),
        step=0.1,
        initial_states=initial_states,
        t=100,
        fname="model_phase_portrait.svg",
    )

    print("Testing visualizations on random matrices...")
    random_matrices = [PRNG.random(size=(3, 3)) / 2.0 for _ in range(100)]
    eigvals = compute_eigenvalues(random_matrices)

    plot_eigenvalues(eigvals, fname="random_matrices.svg")
    plot_Astar_anim(random_matrices, eigvals, fname="random_matrices.mp4")

    print("Testing visualizations on model...")
    gen_model_visuals(
        model,
        SYSTEM,
        eigval_plot_fname="test_model_eigvals.svg",
        anim_fname="test_model.mp4",
    )
