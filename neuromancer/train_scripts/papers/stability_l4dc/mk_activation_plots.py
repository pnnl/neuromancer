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

def plot_act(x_vec, g, color=[1., 0.25, 0.25], fname=None):
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X_stable)):
        axe.plot(x_vec, g, c=color, linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel('$g(z)$', fontsize=22)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close()


if __name__ == '__main__':

    outdir = "plots_20201230_activations"
    os.makedirs(outdir, exist_ok=True)

    x_line = torch.range(-6, 6, step=0.01)
    x_line = x_line[x_line != 0.0]  # drop 0 due to division by 0
    x_vec = x_line.detach().numpy()

    activations = [torch.nn.Identity()]
    for name, act in nact.activations.items():
        # print(name)
        f = act()
        activations.append(f)

    X_stable = []
    X_BIBO_stable = []
    X_unstable = []
    Act_stable = []
    Act_BIBO_stable = []
    Act_unstable = []

    for act in activations:
        x = act(x_line).detach().numpy()
        if any(abs(x/x_vec) > 1):
            # x_1 = act(torch.tensor([5.0])).detach().numpy()
            # x_2 = act(torch.tensor([6.0])).detach().numpy()
            # x_3 = act(torch.tensor([-5.0])).detach().numpy()
            # x_4 = act(torch.tensor([-6.0])).detach().numpy()
            # if abs(x_1 - x_2).squeeze() <= 0.01 and abs(x_3 - x_4).squeeze() <= 0.01:
            x_1 = act(torch.tensor([-6.0])).detach().numpy()
            x_2 = act(torch.tensor([6.0])).detach().numpy()
            if abs(x_1).squeeze() < 6.0 and abs(x_2).squeeze() < 6.0:
                X_BIBO_stable.append(x)
                Act_BIBO_stable.append(act)
            else:
                X_unstable.append(x)
                Act_unstable.append(act)
        else:
            X_stable.append(x)
            Act_stable.append(act)
        plot_act(x_vec, x, fname=os.path.join(outdir, f"act_{str(act)}.png"))

    print(f'unstable activations: {Act_unstable}')
    print(f'stable activations: {Act_stable}')

    # Stable activations
    rgb_colors = get_colors(len(X_stable)+1)
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
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel('$g(z)$', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"act_stable.png"))


    # Unstable activations
    rgb_colors = get_colors(len(X_unstable)+1)
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
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel('$g(z)$', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"act_unstable.png"))

    # BIBO stable activations
    rgb_colors = get_colors(len(X_BIBO_stable)+1)
    plt.style.use('classic')
    fig, (axe) = plt.subplots(1, 1)
    axe.set_ylim(-6, 6)
    axe.set_xlim(-6, 6)
    axe.set_aspect(1)
    axe.fill_between(x_vec, x_vec, -x_vec, alpha=0.1)
    for k in range(len(X_BIBO_stable)):
        axe.plot(x_vec, X_BIBO_stable[k], c=rgb_colors[k], linewidth=2)
    axe.tick_params(axis='x', labelsize=18)
    axe.tick_params(axis='y', labelsize=18)
    axe.grid(True)
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel('$g(z)$', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"act_bibo_stable.png"))



