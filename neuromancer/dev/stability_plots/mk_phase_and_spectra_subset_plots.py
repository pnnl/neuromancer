import os

import torch
from torch import nn
import numpy as np
import slim
from neuromancer import blocks
import matplotlib.pyplot as plt

from lpv import lpv
from phase_plots import plot_model_streamlines
from eigen_plots import compute_eigenvalues, plot_eigenvalues, plot_matrix_eigval_anim


if __name__ == "__main__":
    SEED = 410
    outdir = "plots_20201120"
    plt.rcParams["figure.dpi"] = 300
    nx = 2

    os.makedirs(outdir, exist_ok=True)

    # initial states used for plotting trajectories in phase portraits
    initial_states = torch.from_numpy(
        np.array(
            [
                [4 * np.cos(x), 4 * np.sin(x)]
                for x in np.arange(0.25, 2.25 * np.pi, np.pi / 4)
            ]
        )
    ).float()

    activations = {
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus
    }

    combos = [
        # chaotic attractors
        (("softSVD", 0.99, 1.0, False), 8, [("selu", False)]),
        (("softSVD", 0.99, 1.01, False), 8, [("selu", False)]),
        (("softSVD", 0.99, 1.1, False), 8, [("selu", False)]),
        (("softSVD", 1.0, 1.01, False), 8, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 8, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 8, [("tanh", False)]),


        # attractor demonstration
        (("softSVD", 0.99, 1.0, False), 4, [("selu", False)]),
        (("softSVD", 0.99, 1.0, False), 8, [("selu", False)]),
        (("gershgorin", 0.0, 1.0, True), 1, [("relu", False)]),
        (("gershgorin", -1.5, -1.1, True), 1, [("selu", True)]),
        (("pf", 1.0, 1.0, False), 1, [("relu", False)]),
        (("softSVD", 0.99, 1.1, False), 1, [("selu", False)]),
        (("gershgorin", 0.99, 1.1, False), 1, [("softplus", False)]),

        # spectrum demonstration
        (("gershgorin", 0.0, 1.0, False), 1, [("gelu", False)]),
        (("gershgorin", 0.0, 1.0, False), 4, [("gelu", False)]),
        (("gershgorin", 0.0, 1.0, False), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, False), 1, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, False), 4, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, False), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.1, True), 1, [("gelu", True)]),
        (("gershgorin", 0.99, 1.1, True), 4, [("gelu", True)]),
        (("gershgorin", 0.99, 1.1, True), 8, [("gelu", True)]),

        # bias demonstration
        (("gershgorin", 0.0, 1.0, False), 1, [
            ("relu", True),
            ("relu", False),
        ]),

        # dynamics demonstration
        (("gershgorin", 0.0, 1.0, False), 8, [("selu", False)]),
        (("softSVD", 0.99, 1.1, False), 8, [("tanh", False)]),
        (("gershgorin", 1.0, 1.01, True), 8, [("softplus", True)]),
    ]

    for (linmap, sigmin, sigmax, real), nlayers, params in combos:
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

            if nx == 2:
                plot_model_streamlines(
                    fx,
                    x_lims=(-6, 6),
                    y_lims=(-6, 6),
                    step=0.5,
                    use_bias=bias,
                    t=20,
                    initial_states=initial_states,
                    fname=os.path.join(outdir, f"phase_{combo_string}.pdf"),
                )

                grid_x, grid_y = torch.meshgrid(
                    torch.arange(-6, 6, 0.5),
                    torch.arange(-6, 6, 0.5),
                )
                X = torch.stack((grid_x.flatten(), grid_y.flatten())).T
            else:
                X = torch.arange(-6, 6, 0.5).unsqueeze(-1).expand(-1, nx)

            Astars = []
            for x in X:
                Astar, Astar_b, *_ = lpv(fx, x)
                Astars += [Astar_b.detach().cpu().numpy() if bias else Astar.detach().cpu().numpy()]
            eigvals = compute_eigenvalues(Astars)

            # plot_eigenvalues(eigvals, fname=os.path.join(outdir, f"spectrum_{combo_string}.pdf"))
            # plot_matrix_eigval_anim(Astars, eigvals, fname=os.path.join(outdir, f"spectrum_{combo_string}.mp4"))
