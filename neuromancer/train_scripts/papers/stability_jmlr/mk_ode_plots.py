import slim
import psl
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    u = np.zeros((1, ode.U.shape[-1])) if hasattr(ode, "U") else None

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

    if len(initial_states) > 0:
        states = np.empty((t + 1, *initial_states.shape))
        print(states.shape)
        states[0, :, :] = initial_states
        for n in range(initial_states.shape[0]):
            for i in range(t):
                states[i + 1, n, :] = ode.simulate(nsim=1, x0=states[i, n, :], U=None)["Y"][0, :]
        states = np.transpose(states, axes=(0, 2, 1))
        ax.plot(states[:, 0], states[:, 1], marker="o", ms=3)

    ax.set_xlim((x_lims[0] - step / 2, x_lims[1] - step / 2))
    ax.set_ylim((y_lims[0] - step / 2, y_lims[1] - step / 2))

    if fname is not None:
        plt.savefig(fname)

    return [ax]


if __name__ == "__main__":
    # initial states used for plotting trajectories in phase portraits
    initial_states = np.array(
        [
            [np.cos(x), np.sin(x)]
            for x in np.arange(0.25, np.pi / 2, np.pi / 16)
        ]
    )

    # TODO(lltt): find good parameters, initial states for this one
    system = psl.Pendulum()
    plot_ode_phase_portrait(
        system,
        x_lims=(-4, 4),
        y_lims=(-4, 4),
        step=0.25,
        initial_states=np.array(
            [
                [np.cos(x), np.sin(x)]
                for x in np.arange(0., 2 * np.pi, np.pi / 2)
            ]
        )
    )
    plt.show()

    system = psl.LotkaVolterra()
    plot_ode_phase_portrait(
        system,
        x_lims=(0, 100),
        y_lims=(0, 100),
        step=10,
        initial_states=np.array(
            [
                [np.cos(x), np.sin(x)]
                for x in np.arange(0.25, np.pi / 2, np.pi / 16)
            ]
        )
    )
    plt.show()

    # TODO(lltt): different system? double pendulum?
    system = psl.Duffing()
    plot_ode_phase_portrait(
        system,
        x_lims=(-1, 2),
        y_lims=(-4, 4),
        step=0.3,
        initial_states=np.array(
            [
                [np.cos(x), np.sin(x)]
                for x in np.arange(0.25, 2 * np.pi, np.pi / 4)
            ]
        ) * 0.1
    )
    plt.show()
