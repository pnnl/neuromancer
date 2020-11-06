import torch
import numpy as np
import matplotlib.pyplot as plt
import dill

from neuromancer.datasets import EmulatorDataset
from psl.nonautonomous import TwoTank, CSTR

from sklearn.decomposition import PCA


def plot_ode_phase_portrait(
    ode,
    x_lims=(0, 1),
    y_lims=(0, 1),
    step=0.05,
    data=None,
    initial_states=[],
    t=100,
    fname=None,
):
    """
    generate phase portrait from ground-truth ODE.

    accepts a psl ODE which will be integrated for one step using a grid of
    initial conditions determined by `x_lims`, `y_lims`, and `step`.

    TODO: if `initial_states` is specified, trajectories over `t` steps will
    be plotted over portrait.
    """

    X0, X1 = np.meshgrid(np.arange(*x_lims, step), np.arange(*y_lims, step))
    X = np.stack((X0.flatten(), X1.flatten())).T
    u = np.zeros((1, ode.U.shape[-1]))

    x_plus = []
    for x in X:
        x_plus += [ode.simulate(nsim=1, x0=x, U=u)["Y"][0, :]]  # , U=U)['Y']]
    x_plus = np.stack(x_plus)

    Xp0, Xp1 = np.stack(x_plus).T.reshape(2, *X0.shape)

    Xp0 = -X0 + Xp0
    Xp1 = -X1 + Xp1

    # plot vector field
    _, ax = plt.subplots()
    ax.quiver(
        X0,
        X1,
        Xp0,
        Xp1,
        angles="uv",
        pivot="mid",
        width=0.002,
        headwidth=4,
        headlength=5,
    )

    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))

    if fname is not None:
        plt.savefig(fname)

    return [ax]


class SsmModel:
    def __init__(self, model, use_input=False):
        self.estim = model.components[0]
        self.fx = model.components[1].fx
        self.fu = model.components[1].fu if use_input else lambda x: 0.0
        self.fy = model.components[1].fy

    def __call__(self, x, u=None):
        x = {"Yp": x.expand(self.estim.nsteps, -1, -1)}
        x = self.estim(x)["x0_estim"]
        x = self.fx(x) + self.fu(u)
        x = self.fy(x)
        return x


def plot_model_phase_portrait(
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
    X, Y = torch.meshgrid(torch.arange(*x_lims, step), torch.arange(*y_lims, step))
    grid = torch.stack((X.flatten(), Y.flatten())).T
    grid = model(grid)

    U, V = grid.T.reshape(2, *X.shape).detach().cpu().numpy()

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
        initial_states = torch.tensor(
            initial_states, dtype=torch.float, requires_grad=False
        )
        states = torch.empty(t + 1, *initial_states.shape, requires_grad=False)
        states[0, :, :] = initial_states
        for i in range(t):
            states[i + 1, :, :] = model(states[i, :, :])
        states = states.transpose(1, 2).detach().cpu().numpy()
        ax.plot(states[:, 0], states[:, 1])

    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))

    if fname is not None:
        plt.savefig(fname)

    return [ax]


def plot_phase_portrait_hidim(fx, limits=(-1, 1), nsamples=10000, fname=None):
    """
    NOTE:
    this isn't very useful, just an initial attempt at plotting portraits in
    high-dimensional space using PCA before settling on processing state
    samples with state estimator and SSM.
    """
    x = torch.rand(nsamples, nx, dtype=torch.float)
    z = fx(x)
    z = z.detach().cpu().numpy()

    xfm = PCA(n_components=2)
    x = xfm.fit_transform(x)
    x, z = xz[:nsamples], xz[nsamples:]

    X, Y = x.T
    U, V = z.T

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(X, Y, -U, -V, angles="xy")

    return [ax]


if __name__ == "__main__":
    print("Testing ODE phase portrait...")
    plot_ode_phase_portrait(CSTR(nsim=1))
    plt.show()

    plot_ode_phase_portrait(TwoTank(nsim=1))
    plt.show()

    CSTR_PATH = "neuromancer/train_scripts/L4DC_paper/models/cstr_model.pth"
    TANK_PATH = "neuromancer/train_scripts/L4DC_paper/models/tank_model.pth"

    cstr_model = torch.load(CSTR_PATH, pickle_module=dill, map_location="cpu")
    cstr_data = EmulatorDataset("CSTR", nsim=10000, seed=50, device="cpu")

    tank_model = torch.load(TANK_PATH, pickle_module=dill, map_location="cpu")
    tank_data = EmulatorDataset("TwoTank", nsim=10000, seed=81, device="cpu")

    initial_states = [
        np.array([0.5, 0.25]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.75]),
    ]

    print("Testing model phase portrait...")

    plot_model_phase_portrait(
        SsmModel(cstr_model),
        data=cstr_data.train_loop,
        initial_states=initial_states,
    )
    plt.show()

    plot_model_phase_portrait(
        SsmModel(tank_model),
        data=tank_data.train_loop,
        initial_states=initial_states,
    )
    plt.show()

    initial_states = [
        np.array([-2, 0]),
        np.array([0.5, -1]),
    ]
    spiral_A = torch.tensor([[0, 0.8], [-0.05, -0.0135]], dtype=torch.float).T
    plot_model_phase_portrait(
        lambda x: torch.matmul(x, spiral_A),
        x_lims=(-2, 2),
        y_lims=(-2, 2),
        step=0.2,
        initial_states=initial_states,
        t=100,
    )
    plt.show()