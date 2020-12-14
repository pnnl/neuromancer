# python base imports
import argparse
import dill
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.linalg as LA
import os
# machine learning data science imports
import numpy as np
import torch
import matplotlib.patches as mpatches
import time


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
        elif hasattr(model.fx, 'rnn'):
            if isinstance(model.fx.rnn, torch.nn.RNN):
                rows = len(model.fx.rnn.all_weights[0])
                Mat = []
                for cell in model.fx.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(model.fx.rnn.rnn_cells)
                Mat = []
                for cell in model.fx.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
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
        elif hasattr(model.fxud, 'rnn'):
            if isinstance(dynamics.fxud.rnn, torch.nn.RNN):
                rows = len(dynamics.fxud.rnn.all_weights[0])
                Mat = []
                for cell in dynamics.fxud.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(dynamics.fxud.rnn.rnn_cells)
                Mat = []
                for cell in dynamics.fxud.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')
    print(rows)
    if rows == 1:
        fig, (eigax1) = plt.subplots(rows, 1)
        eigax1.set_ylim(-1.1, 1.1)
        eigax1.set_xlim(-1.1, 1.1)
        eigax1.set_aspect(1)
        if not mat.shape[0] == mat.shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(mat.T)
            w = np.sqrt(w)
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
            ro = int(np.floor(k / 2))
            if len(axes.shape) > 1:
                axes[ro, 0].set_ylim(-1.1, 1.1)
                axes[ro, 0].set_xlim(-1.1, 1.1)
                axes[ro, 0].set_aspect(1)
                axes[ro, 1].set_ylim(-1.1, 1.1)
                axes[ro, 1].set_xlim(-1.1, 1.1)
                axes[ro, 1].set_aspect(1)
            else:
                axes[k].set_ylim(-1.1, 1.1)
                axes[k].set_xlim(-1.1, 1.1)
                axes[k].set_aspect(1)
                axes[k].set_ylim(-1.1, 1.1)
                axes[k].set_xlim(-1.1, 1.1)
                axes[k].set_aspect(1)
            if not Mat[k].shape[0] == Mat[k].shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(Mat[k].T)
                w = np.sqrt(w)
                # axes[int(np.floor(k / 2)), 0].set_title('Singular values')
                # axes[int(np.floor(k / 2)), 1].set_title('Singular values')
            else:
                w, v = LA.eig(Mat[k].T)
                # axes[int(np.floor(k / 2)), 0].set_title('Eigenvalues') if count == 0 else None
                # axes[int(np.floor(k / 2)), 1].set_title('Eigenvalues') if count == 0 else None
                count += 1
            if k % 2 == 0:
                if len(axes.shape) > 1:
                    axes[int(np.floor(k / 2)), 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                    axes[int(np.floor(k / 2)), 0].add_patch(patch)
                else:
                    axes[k].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                    axes[k].add_patch(patch)
            else:
                if len(axes.shape) > 1:
                    axes[int(np.floor(k / 2)), 1].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                    axes[int(np.floor(k / 2)), 1].add_patch(patch)
                else:
                    axes[k].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                    axes[k].add_patch(patch)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))



def plot_eigenvalues_compact(model, savedir='./test/'):
    Mat = []
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            Mat.append(model.fx.effective_W().detach().cpu().numpy())
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        elif hasattr(model.fx, 'rnn'):
            if isinstance(model.fx.rnn, torch.nn.RNN):
                rows = len(model.fx.rnn.all_weights[0])
                for cell in model.fx.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(model.fx.rnn.rnn_cells)
                for cell in model.fx.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            Mat.append(model.fxud.effective_W().detach().cpu().numpy())
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        elif hasattr(model.fxud, 'rnn'):
            if isinstance(dynamics.fxud.rnn, torch.nn.RNN):
                rows = len(dynamics.fxud.rnn.all_weights[0])
                for cell in dynamics.fxud.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(dynamics.fxud.rnn.rnn_cells)
                for cell in dynamics.fxud.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')
    W = []
    for k in range(rows):
        if not Mat[k].shape[0] == Mat[k].shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(Mat[k].T)
            w = np.sqrt(w)
        else:
            w, v = LA.eig(Mat[k].T)
        W = np.append(W, w)
    fig, (eigax1) = plt.subplots(1, 1)
    eigax1.set_ylim(-1.1, 1.1)
    eigax1.set_xlim(-1.1, 1.1)
    eigax1.set_aspect(1)
    eigax1.scatter(W.real, W.imag, alpha=0.5, c=get_colors(len(W.real)))
    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
    eigax1.add_patch(patch)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'eigmat.png'))


if __name__ == '__main__':

    paths = ['../models/best_models_bnl_paper/blocknlin/aero/best_model.pth',
             '../models/best_models_bnl_paper/blocknlin/cstr/best_model.pth',
             '../models/best_models_bnl_paper/blocknlin/twotank/best_model.pth',
             '../models/nonlin_ablation_models/aero_uberablate_best_model.pth',
             '../models/nonlin_ablation_models/CSTR_uberablate_best_model.pth',
             '../models/nonlin_ablation_models/TwoTank_uberablate_best_model.pth']
             # '../models/best_models_bnl_paper/blackbox/aero/best_model.pth',
             # '../models/best_models_bnl_paper/blackbox/cstr/best_model.pth',
             # '../models/best_models_bnl_paper/blackbox/twotank/best_model.pth']

    device = torch.device('cpu')
    for path in paths:
        model = torch.load(path, pickle_module=dill, map_location=device)
        dynamics = model.components[1]
        # plot_eigenvalues(dynamics)
        plot_eigenvalues_compact(dynamics)