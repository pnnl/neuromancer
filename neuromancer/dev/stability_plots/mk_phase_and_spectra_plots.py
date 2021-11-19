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


def phase_and_spectra_plot_loop(nx, layers, maps, activations, outdir="plots_20201117"):
    for nlayers in layers:
        for (linmap, sigmin, sigmax, real) in maps:
            torch.manual_seed(408)
            np.random.seed(408)
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
            for (act_name, act) in activations:
                fx.nonlin = nn.ModuleList([act() for _ in fx.nonlin])
                for bias in [True, False]:
                    combo_string = f"{linmap}_x{nx}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}{'_bias' if bias else ''}"
                    print(combo_string)
                    if not os.path.exists(os.path.join(outdir, f"phase_{combo_string}.png")):
                        if nx == 2:
                            plot_model_streamlines(
                                fx,
                                x_lims=(-6, 6),
                                y_lims=(-6, 6),
                                step=0.5,
                                use_bias=bias,
                                initial_states=initial_states,
                                fname=os.path.join(outdir, f"phase_{combo_string}.png"),
                            )

                    if not os.path.exists(os.path.join(outdir, f"spectrum_{combo_string}.png")):
                        if nx == 2:
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

                        plot_eigenvalues(eigvals, fname=os.path.join(outdir, f"spectrum_{combo_string}.png"))
                    # plot_matrix_eigval_anim(Astars, eigvals, fname=os.path.join(outdir, f"spectrum_{combo_string}.mp4"))
                        plt.close('all')


if __name__ == "__main__":
    outdir = "plots_all_20201118"
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

    # grid search components
    maps = [
        ("softSVD", -1.5, -1.1, False),
        ("softSVD", 0.0, 1.0, False),
        ("softSVD", 0.99, 1.0, False),
        ("softSVD", 0.99, 1.1, False),
        ("softSVD", 1.0, 1.01, False),
        ("softSVD", 0.99, 1.01, False),
        ("softSVD", 1.1, 1.5, False),
        ("gershgorin", -1.5, -1.1, True),
        ("gershgorin", 0.0, 1.0, True),
        ("gershgorin", 0.99, 1.0, True),
        ("gershgorin", 0.99, 1.1, True),
        ("gershgorin", 1.0, 1.01, True),
        ("gershgorin", 0.99, 1.01, True),
        ("gershgorin", 1.1, 1.5, True),
        ("gershgorin", -1.5, -1.1, False),
        ("gershgorin", 0.0, 1.0, False),
        ("gershgorin", 0.99, 1.0, False),
        ("gershgorin", 0.99, 1.1, False),
        ("gershgorin", 1.0, 1.01, False),
        ("gershgorin", 0.99, 1.01, False),
        ("gershgorin", 1.1, 1.5, False),
        ("pf", 1.0, 1.0, False),
        ("linear", 1.0, 1.0, False),
    ]
    activations = [
        ("relu", torch.nn.ReLU),
        ("selu", torch.nn.SELU),
        ("gelu", torch.nn.GELU),
        ("tanh", torch.nn.Tanh),
        ("sigmoid", torch.nn.Sigmoid),
        ("softplus", torch.nn.Softplus)
    ]
    layers = [8, 4, 1]

    # phase_and_spectra_plot_loop(10, layers, maps, activations)
    phase_and_spectra_plot_loop(2, layers, maps, activations, outdir=outdir)
