import matplotlib.pyplot as plt
import numpy as np


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

def pltOL_train(Ytrue, Ytrain, X=None, U=None, D=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """
    nrows = 4
    if U is None:
        nrows -= 1
    if D is None:
        nrows -= 1

    fig, ax = plt.subplots(nrows, 1, figsize=(20, 16))

    if nrows == 1:
        ax.plot(Ytrue, linewidth=3)
        ax.plot(Ytrain, '--', linewidth=3)
        ax.grid(True)
        ax.set_title('Outputs', fontsize=24)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Y', fontsize=24)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
    else:
        ax[0].plot(Ytrue, linewidth=3)
        ax[0].plot(Ytrain, '--', linewidth=3)
        ax[0].grid(True)
        ax[0].set_title('Outputs', fontsize=24)
        ax[0].set_xlabel('Time', fontsize=24)
        ax[0].set_ylabel('Y', fontsize=24)
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)

    if X is not None:
        ax[1].plot(X, linewidth=3)
        ax[1].grid(True)
        ax[1].set_title('States', fontsize=24)
        ax[1].set_xlabel('Time', fontsize=24)
        ax[1].set_ylabel('X', fontsize=24)
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)

    if U is not None:
        idx = 2
        if X is None:
            idx -= 1
        ax[idx].plot(U, linewidth=3)
        ax[idx].grid(True)
        ax[idx].set_title('Inputs', fontsize=24)
        ax[idx].set_xlabel('Time', fontsize=24)
        ax[idx].set_ylabel('U', fontsize=24)
        ax[idx].tick_params(axis='x', labelsize=22)
        ax[idx].tick_params(axis='y', labelsize=22)

    if D is not None:
        idx = 3
        if U is None:
            idx -= 1
        if X is None:
            idx -= 1
        ax[idx].plot(D, linewidth=3)
        ax[idx].grid(True)
        ax[idx].set_title('Disturbances', fontsize=24)
        ax[idx].set_xlabel('Time', fontsize=24)
        ax[idx].set_ylabel('D', fontsize=24)
        ax[idx].tick_params(axis='x', labelsize=22)
        ax[idx].tick_params(axis='y', labelsize=22)



def pltOL(Y, U=None, D=None, X=None):
    """
    plot input output open loop dataset
    """
    nrows = 4
    if X is None:
        nrows -= 1
    if U is None:
        nrows -= 1
    if D is None:
        nrows -= 1

    fig, ax = plt.subplots(nrows, 1, figsize=(20, 16))

    if nrows == 1:
        ax.plot(Y, linewidth=3)
        ax.grid(True)
        ax.set_title('Outputs', fontsize=24)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Y', fontsize=24)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
    else:
        ax[0].plot(Y, linewidth=3)
        ax[0].grid(True)
        ax[0].set_title('Outputs', fontsize=24)
        ax[0].set_xlabel('Time', fontsize=24)
        ax[0].set_ylabel('Y', fontsize=24)
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)

    if X is not None:
        ax[1].plot(X, linewidth=3)
        ax[1].grid(True)
        ax[1].set_title('States', fontsize=24)
        ax[1].set_xlabel('Time', fontsize=24)
        ax[1].set_ylabel('X', fontsize=24)
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)

    if U is not None:
        idx = 2
        if X is None:
            idx -= 1
        ax[idx].plot(U, linewidth=3)
        ax[idx].grid(True)
        ax[idx].set_title('Inputs', fontsize=24)
        ax[idx].set_xlabel('Time', fontsize=24)
        ax[idx].set_ylabel('U', fontsize=24)
        ax[idx].tick_params(axis='x', labelsize=22)
        ax[idx].tick_params(axis='y', labelsize=22)

    if D is not None:
        idx = 3
        if U is None:
            idx -= 1
        if X is None:
            idx -= 1
        ax[idx].plot(D, linewidth=3)
        ax[idx].grid(True)
        ax[idx].set_title('Disturbances', fontsize=24)
        ax[idx].set_xlabel('Time', fontsize=24)
        ax[idx].set_ylabel('D', fontsize=24)
        ax[idx].tick_params(axis='x', labelsize=22)
        ax[idx].tick_params(axis='y', labelsize=22)



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