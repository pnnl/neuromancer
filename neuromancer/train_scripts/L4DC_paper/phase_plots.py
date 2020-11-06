import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA


def plot_ode_phase_portrait(
    x_lims=(0, 1),
    y_lims=(0, 1),
    step=0.02,
    data=None,
    initial_states=[],
    t=100,
    fname=None,
):
    pass


def plot_phase_portrait_2d(
    model,
    x_lims=(0, 1),
    y_lims=(0, 1),
    step=0.02,
    data=None,
    initial_states=[],
    t=100,
    fname=None,
):
    estim = model.components[0]
    fx = model.components[1].fx
    fy = model.components[1].fy

    X, Y = torch.meshgrid(torch.arange(*x_lims, step), torch.arange(*y_lims, step))

    grid = {"Yp": torch.stack((X.flatten(), Y.flatten())).T.unsqueeze(0).expand(estim.nsteps, -1, -1)}
    grid = estim(grid)["x0_estim"]
    grid = fx(grid)
    grid = fy(grid)
    grid = grid.detach().cpu().numpy()

    U, V = grid.reshape(2, *X.shape)

    # TODO(lltt): doesn't seem right to do this
    U = -X + U
    V = -Y + V

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, pivot="mid")

    if data is not None:
        train_states = data["Yp"].squeeze(1).cpu().numpy()
        print(train_states.shape)
        ax.scatter(train_states[:, 0], train_states[:, 1], s=2)

    """
    # plot state trajectories over phase space if initial states given
    initial_states = torch.tensor(initial_states, dtype=torch.float)
    states = torch.empty(t + 1, *initial_states.shape)
    states[0, :, :] = initial_states
    for i in range(t):
        states[i + 1, :, :] = fx(states[i, :, :])
    states = states.transpose(1, 2).detach().cpu().numpy()
    ax.plot(states[:, 0], states[:, 1])
    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))
    if fname is not None:
        plt.savefig(fname)
    """
    return [ax]

def plot_random_phase_portrait(
    model,
    limits=(-1, 1),
    nsamples=10000,
    initial_states=[],
    t=100,
    fname=None,
):

    nx = fx.in_features

    x = torch.rand(nsamples, nx, dtype=torch.float)
    z = fx(x)
    z = z.detach().cpu().numpy()

    xfm = PCA(n_components=2)
    x = xfm.fit_transform(x)
    x, z = xz[:nsamples], xz[nsamples:]

    X, Y = x.T#.reshape(2, -1)
    U, V = z.T#.reshape(2, -1)

    # TODO(lltt): doesn't seem right to do this
    U = -X + U
    V = -Y + V

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V)

    """
    # plot state trajectories over phase space if initial states given
    initial_states = torch.tensor(initial_states, dtype=torch.float)
    states = torch.empty(t + 1, *initial_states.shape)
    states[0, :, :] = initial_states
    for i in range(t):
        states[i + 1, :, :] = fx(states[i, :, :])
    states = states.transpose(1, 2).detach().cpu().numpy()
    ax.plot(states[:, 0], states[:, 1])
    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))
    if fname is not None:
        plt.savefig(fname)
    """

    return [ax]

if __name__ == "__main__":
    # NOTE: will fail for models with nx > 2
    print("Testing model phase portrait...")
    initial_states = [
        # np.array([np.cos(x), np.sin(x)])*0.5 + np.array([0, 0])
        # for x in np.arange(0, 2 * np.pi, np.pi / 2)
        np.array([-2, 0]),
        np.array([0.5, -1]),
    ]
    spiral_A = torch.tensor([[0, 0.8], [-0.05, -0.0135]], dtype=torch.float).T
    plot_phase_portrait(
        lambda x: torch.matmul(x, spiral_A),
        limits=(-2, 2.1),
        step=0.1,
        initial_states=initial_states,
        t=100,
        fname="test_model_phase_portrait.svg",
    )

    plot_random_phase_portrait(
        lambda x: torch.matmul(x, spiral_A),
        limits=(-2, 2.1),
        step=0.1,
        initial_states=initial_states,
        t=100,
        fname="test_model_phase_portrait.svg",
    )
