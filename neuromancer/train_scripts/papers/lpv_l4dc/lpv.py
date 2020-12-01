import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt

import slim
from neuromancer import blocks


def lpv(fx, x):
    """pared-down version of LPV_net"""
    # nonlinearities = fx.nonlin
    x_layer = x
    x_layer_orig = x
    x_layer_Aprime = x

    A_mats = []
    Aprime_mats = []
    activation_mats = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight
        A_mats += [A]
        Ax = torch.matmul(x_layer, A)  # linear transform

        # TODO: if *any* are zero, this will break
        if sum(Ax.squeeze()) == 0:
            lambda_h = torch.zeros(Ax.shape)
        else:
            lambda_h = nlin(Ax) / Ax  # activation scaling

        lambda_h = torch.diag(lambda_h.squeeze())
        activation_mats += [lambda_h]

        x_layer = torch.matmul(Ax, lambda_h)
        x_layer_orig = nlin(lin(x_layer_orig))

        # compute layer-wise parameter-varying linear map
        Aprime = torch.matmul(A, lambda_h)
        x_layer_Aprime = torch.matmul(x_layer_Aprime, Aprime)

        Aprime_mats += [Aprime]

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = torch.chain_matmul(*Aprime_mats)

    return Astar, Aprime_mats, A_mats


def plot_astar_phase_portrait(
    model,
    x_lims=(0, 1),
    y_lims=(0, 1),
    step=0.05,
    data=None,
    initial_states=[],
    t=100,
    fname=None,
):
    """
    generate phase portrait using a model function.

    accepts a function will be evaluated for one step using a grid of initial
    conditions determined by `x_lims`, `y_lims`, and `step`.

    if `initial_states` is specified, trajectories over `t` steps will
    be plotted over portrait.
    """
    X, Y = torch.meshgrid(
        torch.arange(x_lims[0], x_lims[1] + step, step),
        torch.arange(y_lims[0], y_lims[1] + step, step),
    )
    grid = torch.stack((X.flatten(), Y.flatten())).T

    Xplus = torch.empty_like(grid)
    for i, sample in enumerate(grid):
        Astar, _, _ = lpv(model, sample)
        Xplus[i, ...] = torch.matmul(sample, Astar)

    U, V = Xplus.T.reshape(2, *X.shape).detach().cpu().numpy()

    U = -X + U
    V = -Y + V

    # plot vector field
    _, ax = plt.subplots()
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
        ax.plot(states[:, 0], states[:, 1])

    ax.set_xlim((x_lims[0], x_lims[1]))
    ax.set_ylim((y_lims[0], y_lims[1]))
    ax.set_aspect(1)

    if fname is not None:
        plt.savefig(fname)

    return [ax]


if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 300
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

    circles = torch.from_numpy(
        np.array(
            [
                [4 * np.cos(x), 4 * np.sin(x)]
                for x in np.arange(0.25, 2.25 * np.pi, np.pi / 4)
            ]
        )
    ).float()

    for nlayers in [1, 8]:
        for linmap, sigmin, sigmax in square_maps:
            for real in [True, False]:
                for act_name, act in activations:
                    fx = blocks.MLP(
                        nx,
                        nx,
                        nonlin=act,
                        Linear=slim.linear.maps[linmap],
                        hsizes=[nx] * nlayers,
                        bias=False,
                        linargs={
                            "sigma_min": sigmin,
                            "sigma_max": sigmax,
                            "real": real,
                        },
                    )
                    plot_astar_phase_portrait(
                        fx,
                        x_lims=(-6, 6),
                        y_lims=(-6, 6),
                        step=0.5,
                        initial_states=circles,
                        fname=f"{linmap}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}.png",
                    )
