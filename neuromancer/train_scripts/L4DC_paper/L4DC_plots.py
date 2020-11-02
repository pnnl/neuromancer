import torch.nn as nn
import torch
import slim
import scipy.linalg as LA
import dill
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.linalg as LA
import os
import numpy as np
import matplotlib.patches as mpatches



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



if __name__ == '__main__':

    x_vec = torch.range(-6, 6, step=0.01)
    activations = [nn.ReLU(), nn.PReLU(), nn.LeakyReLU(), nn.GELU(), nn.ELU(),
                   nn.CELU(), nn.Softshrink(), nn.Softsign(), nn.Tanhshrink(),
                   nn.Tanh(), nn.Hardshrink(), nn.Hardtanh(), nn.Hardswish()]
    activations_unstable = [nn.SELU()]
    activations_1 = [nn.Sigmoid(), nn.Hardsigmoid()]
    activations_off = [nn.LogSigmoid(), nn.Softplus()]

    # Stable activations
    X = []
    for act in activations:
        x = act(x_vec).detach().numpy()
        X.append(x)
    rgb_colors = get_colors(len(X))
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X)):
        axe.plot(x_vec, X[k], c=rgb_colors[k], linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.tight_layout()

    # Unstable activations
    X = []
    for act in activations_unstable:
        x = act(x_vec).detach().numpy()
        X.append(x)
    rgb_colors = get_colors(len(X))
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X)):
        axe.plot(x_vec, X[k], c=rgb_colors[k], linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.tight_layout()




