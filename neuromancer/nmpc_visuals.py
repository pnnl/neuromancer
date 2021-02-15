"""

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# local imports
from neuromancer.datasets import unbatch_data, unbatch_mh_data
import neuromancer.plot as plot


class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()



class VisualizerClosedLoop2(Visualizer):

    def __init__(self, dataset, policy, plot_keys, verbosity, savedir='test_control'):
        self.model = policy
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys
        self.savedir = savedir

    def plot_traj(self, true_traj, pred_traj, figname='open_loop.png'):
        fig, ax = plt.subplots(len(true_traj), 1)
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            axe = ax if len(true_traj) == 1 else ax[row]
            axe.set_ylabel(label, rotation=0, labelpad=20)
            axe.plot(t1, label='True', c='c')
            axe.plot(t2, label='Pred', c='m')
            axe.tick_params(labelbottom=False)
        axe.tick_params(labelbottom=True)
        axe.set_xlabel('Time')
        axe.legend()
        plt.savefig(figname)

    def eval(self, outputs):
        """

        :param outputs:
        :return:
        """
        for k in ['train_model_', 'train_plant_', 'model_', 'plant_']:
            if f'{k}Y' in outputs.keys():
                D = outputs[f'{k}D'] if f'{k}D' in outputs.keys() else None
                R = outputs[f'{k}R'] if f'{k}R' in outputs.keys() else None
                Ymin = outputs[f'{k}Ymin'] if f'{k}Ymin' in outputs.keys() else None
                Ymax = outputs[f'{k}Ymax'] if f'{k}Ymax' in outputs.keys() else None
                Umin = outputs[f'{k}Umin'] if f'{k}Umin' in outputs.keys() else None
                Umax = outputs[f'{k}Umax'] if f'{k}Umax' in outputs.keys() else None
                pltCL(Y=outputs[f'{k}Y'], U=outputs[f'{k}U'], D=None, R=R,
                           Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                           ctrl_outputs=self.dataset.ctrl_outputs,
                           figname=os.path.join(self.savedir, f'CL_{k}control.png'))

                pltCL_aggregate(Y=outputs[f'{k}Y'], U=outputs[f'{k}U'], D=None, R=R,
                      Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                      ctrl_outputs=self.dataset.ctrl_outputs,
                      figname=os.path.join(self.savedir, f'CL_{k}control2.png'))

                # plot.pltCL(Y=outputs[f'{k}Y'], U=outputs[f'{k}U'], D=D, R=R,
                #       Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                #       ctrl_outputs=self.dataset.ctrl_outputs,
                #       figname=os.path.join(self.savedir, f'CL_{k}control.png'))
        return dict()



def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return: (np.array, shape=[k, 3])
    """
    phi = np.linspace(0, 2 * np.pi, k)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def pltCL(Y, R=None, U=None, D=None, X=None, ctrl_outputs=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset

    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]
    ny = Y.shape[1]
    nu = U.shape[1]

    fig, ax = plt.subplots(nrows=ny, ncols=2, figsize=(20, 16), squeeze=False, constrained_layout=True)
    colors = get_colors(Y.shape[1]+U.shape[1])
    for k in range(Y.shape[1]):
        rk = ctrl_outputs.index(k) if ctrl_outputs is not None and k in ctrl_outputs else None
        ax[k, 0].plot(Y[:, k], '-', linewidth=3, c=colors[k]) if Y[:, k] is not None else None
        ax[k, 0].grid(True)
        ax[k, 0].set_ylabel(f'$y_{k+1} [^o C]$', fontsize=14)
        ax[k, 0].tick_params(axis='x', labelsize=12)
        ax[k, 0].tick_params(axis='y', labelsize=12)
        ax[k, 0].plot(Ymin[:, k], '--', linewidth=3, c='k') if Ymin[:, k] is not None else None
        ax[k, 0].plot(Ymax[:, k], '--', linewidth=3, c='k') if Ymax[:, k] is not None else None
        ax[k, 0].set_xlim(0, Y.shape[0])
        ax[k, 0].set_ylim(12, 32)

        ax[k, 1].plot(U[:, k], linewidth=3, c=colors[k])
        ax[k, 1].plot(Umin[:, k], '--', linewidth=3, c='k') if Umin is not None else None
        ax[k, 1].plot(Umax[:, k], '--', linewidth=3, c='k') if Umax is not None else None
        ax[k, 1].set_ylabel(f'$u_{k+1} [l/h]$', fontsize=14)
        ax[k, 1].tick_params(axis='x', labelsize=12)
        ax[k, 1].tick_params(axis='y', labelsize=12)
        ax[k, 1].set_xlim(0, U.shape[0])
        # plt.subplots_adjust(bottom=0.3)
    # plt.tight_layout()
    fig.set_tight_layout(True)
    plt.show()
    if figname is not None:
        plt.savefig(figname)

def pltCL_aggregate(Y, R=None, U=None, D=None, X=None, ctrl_outputs=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset

    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]
    ny = Y.shape[1]
    nu = U.shape[1]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 16), squeeze=False, constrained_layout=True)
    colors = get_colors(Y.shape[1]+U.shape[1])
    for k in range(ny):
        rk = ctrl_outputs.index(k) if ctrl_outputs is not None and k in ctrl_outputs else None
        ax[0, 0].plot(Y[:, k], '-', linewidth=3, c=colors[k]) if Y[:, k] is not None else None
        ax[0, 0].grid(True)
        ax[0, 0].set_ylabel(f'$y_i$', fontsize=14)
        ax[0, 0].tick_params(axis='x', labelsize=12)
        ax[0, 0].tick_params(axis='y', labelsize=12)
        ax[0, 0].plot(Ymin[:, k], '--', linewidth=3, c='k') if Ymin[:, k] is not None else None
        ax[0, 0].plot(Ymax[:, k], '--', linewidth=3, c='k') if Ymax[:, k] is not None else None
        ax[0, 0].set_xlim(0, Y.shape[0])
        # ax[0, 0].set_ylim(12, 32)

        ax[1, 0].plot(U[:, k], linewidth=3, c=colors[k])
        ax[1, 0].plot(Umin[:, 0], '--', linewidth=3, c='k') if Umin is not None else None
        ax[1, 0].plot(np.max(Umax, axis=1), '--', linewidth=3, c='k') if Umax is not None else None
        ax[1, 0].set_ylabel(f'$u_i$', fontsize=14)
        ax[1, 0].tick_params(axis='x', labelsize=12)
        ax[1, 0].tick_params(axis='y', labelsize=12)
        ax[1, 0].set_xlim(0, U.shape[0])
        # plt.subplots_adjust(bottom=0.3)
    # plt.tight_layout()
    fig.set_tight_layout(True)
    plt.show()
    if figname is not None:
        plt.savefig(figname)
