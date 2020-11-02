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
import neuromancer.activations as nact


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

    x_line = torch.range(-6, 6, step=0.01)
    x_line = x_line[x_line != 0.0]  # drop 0 due to division by 0
    x_vec = x_line.detach().numpy()

    activations = []
    for name, act in nact.activations.items():
        # print(name)
        f = act()
        activations.append(f)

    # Stable activations
    X_stable = []
    X_unstable = []
    Act_stable = []
    Act_unstable = []
    for act in activations:
        x = act(x_line).detach().numpy()
        if any(abs(x/x_vec) > 1):
            X_unstable.append(x)
            Act_unstable.append(act)
        else:
            X_stable.append(x)
            Act_stable.append(act)

    print(f'unstable activations: {Act_unstable}')
    print(f'stable activations: {Act_stable}')

    rgb_colors = get_colors(len(X_stable))
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X_stable)):
        axe.plot(x_vec, X_stable[k], c=rgb_colors[k], linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.tight_layout()

    # Unstable activations
    rgb_colors = get_colors(len(X_unstable))
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X_unstable)):
        axe.plot(x_vec, X_unstable[k], c=rgb_colors[k], linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.tight_layout()




