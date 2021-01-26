import os
import sys

import torch
from torch import nn
import numpy as np
import slim
from neuromancer import blocks
import seaborn as sns
sns.set_theme(context="paper", style="darkgrid")
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../stability_l4dc")
)
from lpv import lpv_batched
from eigen_plots import compute_eigenvalues

if __name__ == "__main__":
    SEED = 410
    outdir = "spectral_distributions_20210102"
    plt.rcParams["figure.dpi"] = 100
    # plt.rcParams["font.size"] = 32
    nx = 2

    os.makedirs(outdir, exist_ok=True)

    activations = {
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus
    }

    combos = [
        # spectrum demonstration
        (("gershgorin", 0.0, 1.0, True), 1, [("gelu", False)]),
        (("gershgorin", 0.0, 1.0, True), 4, [("gelu", False)]),
        (("gershgorin", 0.0, 1.0, True), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, True), 1, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, True), 4, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, True), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.1, True), 1, [("gelu", False)]),
        (("gershgorin", 0.99, 1.1, True), 4, [("gelu", False)]),
        (("gershgorin", 0.99, 1.1, True), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.5, True), 1, [("gelu", False)]),
        (("gershgorin", 0.99, 1.5, True), 4, [("gelu", False)]),
        (("gershgorin", 0.99, 1.5, True), 8, [("gelu", False)]),
    ]

    eigvals = []
    plot_nlayers = []
    for i, ((linmap, sigmin, sigmax, real), nlayers, params) in enumerate(combos):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        fx = blocks.MLP(
            nx,
            nx,
            nonlin=nn.Identity,
            linear_map=slim.linear.maps[linmap],
            hsizes=[nx] * nlayers,
            bias=True,
            linargs={
                "sigma_min": sigmin,
                "sigma_max": sigmax,
                "real": real,
            },
        )
        for act_name, bias in params:
            act = activations[act_name]
            fx.nonlin = nn.ModuleList([act() for _ in fx.nonlin])
            combo_string = f"{linmap}_x{nx}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}{'_bias' if bias else ''}"
            print(combo_string)

            grid_x, grid_y = torch.meshgrid(
                torch.arange(-6, 6, 0.1),
                torch.arange(-6, 6, 0.1),
            )
            X = torch.stack((grid_x.flatten(), grid_y.flatten())).T

            with torch.no_grad():
                Astars, *_ = lpv_batched(fx, X)
            Astars = torch.unbind(Astars, dim=0)

            eigvals.append(compute_eigenvalues(Astars))
            plot_nlayers.append(nlayers)

        if i == len(combos) - 1 or (i < len(combos) - 1 and combos[i][0] != combos[i+1][0]):
            plt.figure(figsize=(5, 4))
            for nl, eigval_list in zip(plot_nlayers, eigvals):
                eigval_list = np.array(eigval_list).flatten()
                sns.kdeplot(eigval_list, log_scale=True, label=f"Layers = {nl}")
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(outdir, f"{combo_string}.pdf"))
            plt.gcf().clear()
            eigvals.clear()