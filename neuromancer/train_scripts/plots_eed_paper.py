

# python base imports
import argparse
import dill
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.linalg as LA
import os
from scipy import signal
import matplotlib.pyplot as plt

# machine learning data science imports
import numpy as np
import torch
import matplotlib.patches as mpatches
import time
import random

# local imports
from neuromancer.plot import pltCL, pltOL, get_colors
from neuromancer.datasets import FileDataset
import psl


def pltCL_paper(Y, R=None, U=None, D=None, X=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset
    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='--'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and R is not None:
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(R[:, k], '--', linewidth=3, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k])
                ax[j, 0].legend(custom_lines, ['Reference', 'Output'])
                ax[j, 0].plot(Ymin[:, k], '--', linewidth=3, c='k') if Ymin is not None else None
                ax[j, 0].plot(Ymax[:, k], '--', linewidth=3, c='k') if Ymax is not None else None
        if notation == 'U':
            for k in range(array.shape[1]):
                ax[j, 0].plot(array, linewidth=3)
                ax[j, 0].plot(Umin[:, k], '--', linewidth=3, c='k') if Umin is not None else None
                ax[j, 0].plot(Umax[:, k], '--', linewidth=3, c='k') if Umax is not None else None
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=12)
        ax[j, 0].tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def pltOL_paper(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
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
    custom_lines = [Line2D([0], [0], color='indianred', lw=4, linestyle='--'),
                    Line2D([0], [0], color='cornflowerblue', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            notation = 'Y [cm]'
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=3, c='indianred')
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c='cornflowerblue')
                ax[j, 0].legend(custom_lines, ['True', 'Pred'], fontsize=14)
                ax[j, 0].set(xticks=(0, 2000, 4000, 6000, 8000),
                             xticklabels=('0', '500', '1000', '1500', '2000'))
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        # ax[j, 0].set_title(name, fontsize=16)
        ax[j, 0].set_xlabel('Time [s]', fontsize=16)
        ax[j, 0].set_ylabel(notation, fontsize=16)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
        ax[j, 0].axvspan(0, 3000, facecolor='grey', alpha=0.4, zorder=-100)
        ax[j, 0].axvspan(3000, 6000, facecolor='grey', alpha=0.2, zorder=-100)
        ax[j, 0].margins(0, 0.1)
        ax[j, 0].text(700, -5, '             Train                ',
                   bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': 0.0})
        ax[j, 0].text(3700, -5, '           Validation           ',
                   bbox={'edgecolor': 'none', 'facecolor': 'grey', 'alpha': 0.0})
        ax[j, 0].text(6700, -5, '              Test                ',
                   bbox={'edgecolor': 'none', 'facecolor': 'grey', 'alpha': 0.0})

    # plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def plot_eigenvalues(model, savedir='./test/'):
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            mat = model.fx.effective_W().detach().cpu().numpy()
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            Mat = []
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            mat = model.fxud.effective_W().detach().cpu().numpy()
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            Mat = []
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')

    if rows == 1:
        fig, (eigax1) = plt.subplots(rows, 1)
        eigax1.set_ylim(-1.1, 1.1)
        eigax1.set_xlim(-1.1, 1.1)
        eigax1.set_aspect(1)
        if not mat.shape[0] == mat.shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(mat.T)
            eigax1.set_title('Singular values')
        else:
            w, v = LA.eig(mat.T)
            eigax1.set_title('Eigenvalues')
        eigax1.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
        eigax1.add_patch(patch)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))
    elif rows > 1:
        rows2 = int(np.ceil(rows / 2))
        fig, axes = plt.subplots(rows2, 2)
        # axes[0, 0].set_title('Weights Eigenvalues')
        count = 0
        for k in range(rows):
            axes[int(np.floor(k / 2)), 0].set_ylim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 0].set_xlim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 0].set_aspect(1)
            axes[int(np.floor(k / 2)), 1].set_ylim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 1].set_xlim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 1].set_aspect(1)
            if not Mat[k].shape[0] == Mat[k].shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(Mat[k].T)
                # axes[int(np.floor(k / 2)), 0].set_title('Singular values')
                # axes[int(np.floor(k / 2)), 1].set_title('Singular values')
            else:
                w, v = LA.eig(Mat[k].T)
                # axes[int(np.floor(k / 2)), 0].set_title('Eigenvalues') if count == 0 else None
                # axes[int(np.floor(k / 2)), 1].set_title('Eigenvalues') if count == 0 else None
                count += 1
            if k % 2 == 0:
                axes[int(np.floor(k / 2)), 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[int(np.floor(k / 2)), 0].add_patch(patch)
            else:
                axes[int(np.floor(k / 2)), 1].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[int(np.floor(k / 2)), 1].add_patch(patch)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))


def plot_weights(model, savedir='./test/'):
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            mat = model.fx.effective_W().detach().cpu().numpy()
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            Mat = []
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            mat = model.fxud.effective_W().detach().cpu().numpy()
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            Mat = []
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')
    if rows == 1:
        fig, (eigax, matax) = plt.subplots(rows, 2)
        eigax.set_ylim(-1.1, 1.1)
        eigax.set_xlim(-1.1, 1.1)
        eigax.set_aspect(1)
        matax.axis('off')
        matax.set_title('State Transition Weights')
        matax.imshow(mat.T)
        if not mat.shape[0] == mat.shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(mat.T)
            eigax.set_title('Weights Singular values')
        else:
            w, v = LA.eig(mat.T)
            eigax.set_title('Weights Eigenvalues')
        eigax.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))
    elif rows > 1:
        fig, axes = plt.subplots(rows, 2)
        # axes[0, 0].set_title('Weights Eigenvalues')
        axes[0, 1].set_title('State Transition Weights')
        count = 0
        for k in range(rows):
            axes[k, 0].set_ylim(-1.1, 1.1)
            axes[k, 0].set_xlim(-1.1, 1.1)
            axes[k, 0].set_aspect(1)
            axes[k, 1].axis('off')
            axes[k, 1].imshow(Mat[k].T)
            if not Mat[k].shape[0] == Mat[k].shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(Mat[k].T)
                axes[k, 0].set_title('Weights Singular values')
            else:
                w, v = LA.eig(Mat[k].T)
                axes[k, 0].set_title('Weights Eigenvalues') if count == 0 else None
                count += 1
            axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))


def normalize(M, Mmin=None, Mmax=None):
        """
        :param M: (2-d np.array) Data to be normalized
        :param Mmin: (int) Optional minimum. If not provided is inferred from data.
        :param Mmax: (int) Optional maximum. If not provided is inferred from data.
        :return: (2-d np.array) Min-max normalized data
        """
        Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
        Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M_norm = (M - Mmin) / (Mmax - Mmin)
        return np.nan_to_num(M_norm), Mmin.squeeze(), Mmax.squeeze()


def min_max_denorm(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = M*(Mmax - Mmin) + Mmin
    return np.nan_to_num(M_denorm)


if __name__ == '__main__':

    # normalizations = {'Ymin': -1.3061713953490333, 'Ymax': 32.77003662201578,
    #                   'Umin': -2.1711117, 'Umax': 33.45899931,
    #                   'Dmin': 29.46308055, 'Dmax': 48.97325791}

    device = torch.device('cpu')
    model = torch.load('../../../../deepmpc/results_files/blocknlin_constr_bias_pf_rnn_16/best_model.pth',
                       pickle_module=dill, map_location=device)
    plot_eigenvalues(model)
