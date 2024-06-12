"""
# TODO: stream plots for phase spaces of ODEs
# TODO: generate correlation network - https://python-graph-gallery.com/327-network-from-correlation-matrix/
# TODO: plot information-theoretic measures for time series data - https: // elife - asu.github.io / PyInform / timeseries.html


"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyts.image as pytsimg
import pyts.multivariate.image as pytsmvimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return:
    """
    phi = np.linspace(0, 2 * np.pi, k)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def pltPhase(X, Xtrain=None, figname=None):
    """
    plot phase space for 2D and 3D state spaces

    https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/plot_streamplot.html
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.streamplot.html
    https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.quiver.html
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/
    http://systems-sciences.uni-graz.at/etextbook/sw2/phpl_python.html
    """
    fig = plt.figure()
    if X.shape[1] >= 3:
        ax = fig.add_subplot(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2])
        if Xtrain is not None:
            ax.plot(Xtrain[:, 0], Xtrain[:, 1], Xtrain[:, 2], '--')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
    elif X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1])
        plt.plot(X[0, 0], X[0, 1], 'ro')
        if Xtrain is not None:
            plt.plot(Xtrain[:, 0], Xtrain[:, 1], '--')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
    plt.show()


def pltCorrelate(X, figname=None):
    """
    plot correlation matrices of time series data
    https://realpython.com/numpy-scipy-pandas-correlation-python/
    """
    #  Pearson product-moment correlation coefficients.
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
    C = np.corrcoef(X.T)
    im1 = axes[0, 0].imshow(C)
    axes[0, 0].set_title('Pearson correlation coefficients')
    axes[0, 0].set_xlabel('$X$')
    axes[0, 0].set_ylabel('$X$')
    # covariance matrix
    C = np.cov(X.T)
    im2 = axes[0, 1].imshow(C)
    axes[0, 1].set_title('Covariance matrix')
    axes[0, 1].set_xlabel('$X$')
    axes[0, 1].set_ylabel('$X$')
    #  Spearman correlation coefficient
    rho, pval = stats.spearmanr(X, X)
    C = rho[0:X.shape[1], 0:X.shape[1]]
    im3 = axes[0, 2].imshow(C)
    axes[0, 2].set_title('Spearman correlation coefficients')
    axes[0, 2].set_xlabel('$X$')
    axes[0, 2].set_ylabel('$X$')
    plt.tight_layout()
    plt.show()
    if figname is not None:
        plt.savefig(figname)


def pltRecurrence(X, figname=None):
    """
    plot recurrence of time series data
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_rp.html
    https://pyts.readthedocs.io/en/stable/auto_examples/multivariate/plot_joint_rp.html#sphx-glr-auto-examples-multivariate-plot-joint-rp-py
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_mtf.html
    https://arxiv.org/pdf/1610.07273.pdf
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py
    """
    size = np.ceil(np.sqrt(X.shape[1])).astype(int)
    row_off = size-np.ceil(X.shape[1]/size).astype(int)
    # Recurrence plot
    rp = pytsimg.RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1,X.shape[1]+1):
        row = (np.ceil(i/size)-1).astype(int)
        col = (i-1)%size
        C = X_rp[i-1]
        im = axes[row, col].imshow(C)
        axes[row, col].set_title(f'Recurrence plot x_{i}')
        axes[row, col].set_xlabel('time')
        axes[row, col].set_ylabel('time')
    plt.tight_layout()
    plt.show()

    # joint recurrence plot
    jrp = pytsmvimg.JointRecurrencePlot(threshold='point', percentage=50)
    X_jrp = jrp.fit_transform(X.T.reshape(X.shape[1], 1, -1))
    fig = plt.figure()
    C = X_jrp[0]
    plt.imshow(C)
    plt.title('joint recurrence plot')
    plt.xlabel('time')
    plt.ylabel('time')
    plt.tight_layout()
    plt.show()

    # Markov Transition Field
    mtf = pytsimg.MarkovTransitionField(image_size=100)
    X_mtf = mtf.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1, X.shape[1] + 1):
        row = (np.ceil(i / size) - 1).astype(int)
        col = (i - 1) % size
        C = X_mtf[i - 1]
        axes[row, col].imshow(C)
        axes[row, col].set_title(f'Markov Transition Field x_{i}')
        axes[row, col].set_xlabel('X norm discretized')
        axes[row, col].set_ylabel('X norm discretized')
    plt.tight_layout()
    plt.show()

    # Gramian Angular Fields
    gasf = pytsimg.GramianAngularField(image_size=100, method='summation')
    X_gasf = gasf.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1, X.shape[1] + 1):
        row = (np.ceil(i / size) - 1).astype(int)
        col = (i - 1) % size
        C = X_gasf[i - 1]
        im = axes[row, col].imshow(C)
        axes[row, col].set_title(f'Gramian Angular Fields x_{i}')
        axes[row, col].set_xlabel('X norm discretized')
        axes[row, col].set_ylabel('X norm discretized')
    plt.tight_layout()
    plt.show()


def pltOL(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            colors = get_colors(array.shape[1]+1)
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=2, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=2, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=2)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def pltCL(Y, U=None, D=None, X=None, R=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y':
            if R is not None:
                colors = get_colors(array.shape[1]+1)
                for k in range(array.shape[1]):
                    ax[j, 0].plot(array[:, k], '-', linewidth=2, c=colors[k])
                ax[j, 0].plot(R, '--', linewidth=2, c='black')
                ax[j, 0].legend(custom_lines, ['Ref', 'Y'])
            else:
                ax[j, 0].plot(array, linewidth=2)
            if Ymax is not None:
                ax[j, 0].plot(Ymax, '--', linewidth=2, c='red')
            if Ymin is not None:
                ax[j, 0].plot(Ymin, '--', linewidth=2, c='red')
        else:
            ax[j, 0].plot(array, linewidth=2)
            if notation == 'U' and Umax is not None:
                ax[j, 0].plot(Umax, '--', linewidth=2, c='red')
            if notation == 'U' and Umin is not None:
                ax[j, 0].plot(Umin, '--', linewidth=2, c='red')
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def render_gymnasium(x, env_str, render_mode="rgb_array"):
    """
    Render the state of the environment with
    the state vector x using the Farama environment.
    :param x: sequence of states
    :param env_str: environment string
    :param render_mode: rendering mode
    :return: rendered gif of the state sequence
    """
    try:
        import gymnasium
    except ImportError:
        print("Farama gymnasium not installed.\nTry `pip install gymnasium`")
        return

    # if the user has not changed the environment, don't rebuild it
    if (not hasattr(render_gymnasium, 'env') or
            render_gymnasium.env_str != env_str or
            render_gymnasium.render_mode != render_mode):
        render_gymnasium.env = gymnasium.make(env_str, render_mode=render_mode)
        render_gymnasium.env_str = env_str
        render_gymnasium.render_mode = render_mode
        render_gymnasium.env.reset()
        env = render_gymnasium.env

    env = render_gymnasium.env

    if len(x.shape) > 1:
        frames = []
        for i in range(x.shape[0]):
            env.state[:] = x[i, :]
            frames.append(env.render())
        return frames

    env.state[:] = x[:]
    return env.render()


def plot_acrobot_control(data, save_loc=None):
    """
    Plots the height of the acrobot controller using the loss function.
        cos0, sin0 = y[:,0], y[:,1]
        cos1, sin1 = y[:,2], y[:,3]
        terminal = cos0 + (cos0*cos1 - sin0*sin1)
    Args:
    - data: Dict of state variables. Each key has a ndarray of shape [time, state_variable].
    - save_loc: Optional location to save the plot.
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed.\nTry `pip install torch`")
        return
    data = data.copy()
    # Ensure data is in the correct format and convert tensors to numpy
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].detach().cpu().numpy()
        if len(data[key].shape) == 1:
            data[key] = data[key][np.newaxis, :]
        elif len(data[key].shape) > 2:
            data[key] = data[key].squeeze(0)

    # Calculate the height of the acrobot
    cos0, sin0 = data["yn"][:, 0], data["yn"][:, 1]
    cos1, sin1 = data["yn"][:, 2], data["yn"][:, 3]
    terminal = -(cos0 + (cos0 * cos1 - sin0 * sin1)) + 2.0

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.margins(x=0.1)
    plt.ylim(0, 4)
    plt.xlim(0, 400)

    # Plot the height and an array of ones for the reference
    plt.plot(terminal, label="Height", color='blue')
    plt.plot(np.ones_like(terminal) + 2.0, linestyle="--", label="ref Height", color='red')
    plt.title(
        "Acrobot Height",
        fontsize=24,
    )
    plt.ylabel("Height", fontsize=22)
    plt.xlabel("Time", fontsize=22)
    plt.legend(loc="upper right", fontsize=18)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.margins(x=0.1)
    plt.xlim(0, 400)

    plt.plot(data["U"][:, 0], label="u", color='blue')
    plt.title(
        "Control Input",
        fontsize=24,
    )
    plt.ylabel("Torque", fontsize=22)
    plt.xlabel("Time", fontsize=22)
    plt.legend(loc="upper right", fontsize=18)
    plt.grid(True)

    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    plt.show()


def plot_pendulum_control(data, save_loc=None):
    """
    Plots the pendulum state variables and optional reference trajectories.
    
    Args:
    - data: Dict of state variables. Each key has a ndarray of shape [time, state_variable].
    - ref_data: Optional dict of reference trajectories with matching keys to `data`.
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed.\nTry `pip install torch`")
        return

    data = data.copy()
    for key in data.keys():
        if len(data[key].shape) == 1:
            data[key] = data[key][None, :]
        elif len(data[key].shape) > 2:
            try:
                data[key] = data[key].squeeze(0)
            except ValueError:
                raise ValueError(f"Could not reshape {key} with shape {data[key].shape}")
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))

    # Plot angles
    plt.subplot(3, 1, 1)
    plt.margins(x=0)

    plt.plot(data["yn"][:, 0], label="$cos(\\theta)$", color='cornflowerblue')
    plt.plot(data["yn"][:, 1], label="$sin(\\theta)$", color='tomato')

    # Plot the reference for the angles, sin = 0, cos = 1
    plt.plot(
        np.ones_like(data["yn"][:, 0]),
        linestyle="--", label="ref $cos(\\theta)$",
        color='blue'
    )
    plt.plot(
        np.zeros_like(data["yn"][:, 1]),
        linestyle="--",
        label="ref $sin(\\theta)$",
        color='red'
    )

    plt.title(
        "Angles",
        fontsize=24,
        # fontweight='bold'
    )
    plt.ylabel("Angle", fontsize=22)
    plt.legend(loc="upper right", fontsize=18)
    plt.grid(True)

    # Plot angular velocities
    plt.subplot(3, 1, 2)
    plt.margins(x=0)

    plt.plot(data["yn"][:, 2], label="$\\dot \\theta$", color='blue')
    plt.title(
        "Angular Velocities",
        fontsize=24,
        # fontweight='bold'
    )
    plt.ylabel("$\\frac{rad}{s}$", fontsize=22)
    plt.legend(loc="upper right", fontsize=18)
    plt.grid(True)

    # Plot control input
    plt.subplot(3, 1, 3)
    plt.margins(x=0)

    plt.plot(data["U"][:, 0], label="u", color='blue')
    plt.title(
        "Control Input",
        fontsize=24,
        # fontweight='bold'
    )
    plt.ylabel("Torque", fontsize=22)
    plt.xlabel("Time", fontsize=22)
    plt.grid(True)

    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    plt.show()


def plot_acrobot_sysid(data, save_loc=None):
    """
    Plots the acrobot system identification state variables and optional reference trajectories.

    Args:
    - data: Dict of state variables. Each key has a ndarray of shape [time, state_variable].
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed.\nTry `pip install torch`")
        return

    # Ensure data is in the correct format and convert tensors to numpy
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].detach().cpu().numpy()
        if len(data[key].shape) == 1:
            data[key] = data[key][np.newaxis, :]
        elif len(data[key].shape) > 2:
            data[key] = data[key].squeeze(0)

    plt.figure(figsize=(10, 8))

    # Plot angles
    plt.subplot(3, 1, 1)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, 6))

    plt.plot(data["xsys"][:, 0], label="$cos(\\theta_1)$", color=colors[0])
    plt.plot(data["Y"][:, 0], linestyle="--", label="$ref cos(\\theta_1)$", color=colors[0])
    plt.plot(data["xsys"][:, 1], label="$sin(\\theta_1)$", color=colors[1])
    plt.plot(data["Y"][:, 1], linestyle="--", label="$ref sin(\\theta_1)$", color=colors[1])
    plt.plot(data["xsys"][:, 2], label="$cos(\\theta_2)$", color=colors[2])
    plt.plot(data["Y"][:, 2], linestyle="--", label="$ref cos(\\theta_2)$", color=colors[2])
    plt.plot(data["xsys"][:, 3], label="$sin(\\theta_2)$", color=colors[3])
    plt.plot(data["Y"][:, 3], linestyle="--", label="$ref sin(\\theta_2)$", color=colors[3])

    plt.title("Angles")
    plt.ylabel("Angle")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot angular velocities
    plt.subplot(3, 1, 2)
    plt.plot(data["xsys"][:, 4], label="$\\dot \\theta_1$", color=colors[4])
    plt.plot(data["Y"][:, 4], linestyle="--", label="$ref \\dot \\theta_1$", color=colors[4])
    plt.plot(data["xsys"][:, 5], label="$\\dot \\theta_2$", color=colors[5])
    plt.plot(data["Y"][:, 5], linestyle="--", label="$ref \\dot \\theta_2$", color=colors[5])
    plt.title("Angular Velocities")
    plt.ylabel("$\\frac{rad}{s}$")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot control input
    plt.subplot(3, 1, 3)
    plt.plot(data["U"][:, 0], label="u")
    plt.title("Control Input")
    plt.ylabel("Torque")
    plt.xlabel("Time")
    plt.grid(True)

    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    plt.show()


def plot_pendulum_sysid(data, save_loc=None):
    """
    Plots the pendulum system identification state variables and optional reference trajectories.

    Args:
    - data: Dict of state variables. Each key has a ndarray of shape [time, state_variable].
    """

    # Ensure data is in the correct format and convert tensors to numpy
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].detach().cpu().numpy()
        if len(data[key].shape) == 1:
            data[key] = data[key][np.newaxis, :]
        elif len(data[key].shape) > 2:
            data[key] = data[key].squeeze(0)

    plt.figure(figsize=(10, 8))

    # Plot angles
    plt.subplot(3, 1, 1)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, 3))

    plt.plot(data["xsys"][:, 0], label="$cos(\\theta)$", color=colors[0])
    plt.plot(data["Y"][:, 0], linestyle="--", label="$ref cos(\\theta)$", color=colors[0])
    plt.plot(data["xsys"][:, 1], label="$sin(\\theta)$", color=colors[1])
    plt.plot(data["Y"][:, 1], linestyle="--", label="$ref sin(\\theta)$", color=colors[1])

    plt.title("Angles")
    plt.ylabel("Angle")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot angular velocities
    plt.subplot(3, 1, 2)
    plt.plot(data["xsys"][:, 2], label="$\\dot \\theta$", color=colors[2])
    plt.plot(data["Y"][:, 2], linestyle="--", label="$ref \\dot \\theta$", color=colors[2])
    plt.title("Angular Velocities")
    plt.ylabel("$\\frac{rad}{s}$")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot control input
    plt.subplot(3, 1, 3)
    plt.plot(data["U"][:, 0], label="u")
    plt.title("Control Input")
    plt.ylabel("Torque")
    plt.xlabel("Time")
    plt.grid(True)

    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    plt.show()
