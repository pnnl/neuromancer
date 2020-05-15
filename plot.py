import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return:
    """
    phi = np.linspace(0, 2 * np.pi, k)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def plot_matrices(matrices, labels, figname):
    rows = len(matrices)
    cols = len(matrices[0])
    fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(matrices[i][j])
            axes[i, j].title.set_text(labels[i][j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(figname)


def pltCL_train(Y_GT, Y_train, U, D):
    """
    plot trained open loop dataset
    """
    pass


def pltCL(Y, U, D, R):
    """
    plot input output closed loop dataset
    """
    pass


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
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=3, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=24)
        ax[j, 0].set_xlabel('Time', fontsize=24)
        ax[j, 0].set_ylabel(notation, fontsize=24)
        ax[j, 0].tick_params(axis='x', labelsize=22)
        ax[j, 0].tick_params(axis='y', labelsize=22)
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname)


def plot_trajectories(traj1, traj2, labels, figname):
    fig, ax = plt.subplots(len(traj1), 1, figsize=(12, 12))
    for row, (t1, t2, label) in enumerate(zip(traj1, traj2, labels)):
        if t2 is not None:
            ax[row].plot(t1, label=f'True')
            ax[row].plot(t2, '--', label=f'Pred')
        else:
            ax[row].plot(t1)
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    ylabel=label,
                    xlim=(0, len(t1)))
        ax[row].tick_params(labelbottom=False)
        ax[row].axvspan(2016, 4032, facecolor='grey', alpha=0.25, zorder=-100)
        ax[row].axvspan(4032, 6048, facecolor='grey', alpha=0.5, zorder=-100)
    ax[row].tick_params(labelbottom=True)
    ax[row].set_xlabel('Day')
    ax[0].text(64, 30, '             Train                ',
            bbox={'facecolor': 'white', 'alpha': 0.5})
    ax[0].text(2064, 30, '           Validation           ',
            bbox={'facecolor': 'grey', 'alpha': 0.25})
    ax[0].text(4116, 30, '              Test                ',
               bbox={'facecolor': 'grey', 'alpha': 0.5})
    plt.tight_layout()
    plt.savefig(figname)


def trajectory_movie(true_traj, pred_traj, figname='traj.mp4', freq=1, fps=15, dpi=100):
    plt.style.use('dark_background')
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Trajectory Movie', artist='Matplotlib',
                    comment='Demo')
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=1000)
    fig, ax = plt.subplots(len(true_traj), 1)
    true, pred = [], []
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        ax[row].set(xlim=(0, t1.shape[0]),
                    ylim=(min(t1.min(), t2.min()) - 0.1, max(t1.max(), t2.max()) + 0.1))
        ax[row].set_ylabel(label, rotation=0, labelpad=20)
        t, = ax[row].plot([], [], label='True', c='c')
        p, = ax[row].plot([], [], label='Pred', c='m')
        true.append(t)
        pred.append(p)
        ax[row].tick_params(labelbottom=False)
    ax[row].tick_params(labelbottom=True)
    ax[row].set_xlabel('Time')
    ax[row].legend()
    plt.tight_layout()
    with writer.saving(fig, figname, dpi=dpi):
        for k in range(len(true_traj[0])):
            print(k)
            if k % freq == 0:
                for j in range(len(true_traj)):
                    true[j].set_data(range(k), true_traj[j][:k])
                    pred[j].set_data(range(k), pred_traj[j][:k])
                writer.grab_frame()