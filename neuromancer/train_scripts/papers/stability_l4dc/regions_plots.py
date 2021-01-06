import sys
import os

import torch
from torch import nn
import numpy as np
import slim
from neuromancer import blocks
import matplotlib.pyplot as plt
import seaborn as sns

os.sys.path.append("neuromancer/train_scripts")
os.sys.path.append("neuromancer/train_scripts/stability_l4dc")

from lpv import lpv_batched
from phase_plots import plot_astar_phase_portrait
from eigen_plots import compute_eigenvalues

PALETTE = sns.color_palette(
    "vlag", as_cmap=True
)  # sns.color_palette("crest_r", as_cmap=True)

def compute_norms(matrices):
    m_norms = []
    for m in matrices:
        assert len(m.shape) == 2
        m_norm= np.linalg.norm(m.T)
        m_norms += [m_norm]
    return m_norms

def lin_regions(nx, layers, maps, activations, outdir="./plots_region"):
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
                bias=False,
                linargs={
                    "sigma_min": sigmin,
                    "sigma_max": sigmax,
                    "real": real,
                },
            )
            for (act_name, act) in activations:
                fx.nonlin = nn.ModuleList([act() for _ in fx.nonlin])
                for bias in [False]:
                    combo_string = f"{linmap}_x{nx}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}{'_bias' if bias else ''}"
                    print(combo_string)

                    if nx == 2:
                        plot_astar_phase_portrait(
                            fx,
                            x_lims=(-6, 6),
                            y_lims=(-6, 6),
                            step=0.5,
                            use_bias=bias,
                            initial_states=initial_states,
                            fname=os.path.join(outdir, f"phase_{combo_string}.png"),
                        )

                        grid_x, grid_y = torch.meshgrid(
                            torch.arange(-6, 6.1, 0.1),
                            torch.arange(-6, 6.1, 0.1),
                        )
                        X = torch.stack((grid_x.flatten(), grid_y.flatten())).T
                    else:
                        X = torch.arange(-6, 6.1, 0.1).unsqueeze(-1).expand(-1, nx)

                    Astars, Astar_b, _, _, _ = lpv_batched(fx, X)
                    Astars = Astars.detach().numpy()

                    # plot Anorms
                    Anorms = compute_norms(Astars)
                    Anorm_mat = np.reshape(Anorms, grid_x.shape)

                    fig1, ax1 = plt.subplots()

                    im1 = ax1.imshow(Anorm_mat, vmin=abs(Anorm_mat).min(), vmax=abs(Anorm_mat).max(), cmap=PALETTE, origin='lower',
                               extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
                    fig1.colorbar(im1, ax=ax1)
                    im1.set_clim(0., 2.)

                    ax1.set_title('Metric: 'r'$\Vert A^* \Vert$')

                    fname1 = os.path.join(outdir, f"norm_region_{combo_string}.png")
                    plt.savefig(fname1)

                    # plot dominant eigenvalues
                    eigvals = compute_eigenvalues(Astars)
                    dom_eigs = [np.absolute(eigs).max() for eigs in eigvals]
                    dom_eigs_mat = np.reshape(dom_eigs, grid_x.shape)

                    fig2, ax2 = plt.subplots()

                    vmin = abs(dom_eigs_mat).min()
                    vmax = abs(dom_eigs_mat).max()

                    im2 = ax2.imshow(dom_eigs_mat, vmin=0, vmax=2,
                                     cmap=PALETTE, origin='lower',
                                     extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
                    fig2.colorbar(im2, ax=ax2)
                    im2.set_clim(0., 2.)

                    ax2.set_title('Metric: 'r'$\| \lambda_1 \|$')

                    fname2 = os.path.join(outdir, f"dom_eig_region_{combo_string}.png")
                    plt.savefig(fname2)

                    # plot sum of absolute eigenvalues
                    sum_eigs = [np.absolute(eigs).sum() for eigs in eigvals]
                    sum_eigs_mat = np.reshape(sum_eigs, grid_x.shape)

                    fig3, ax3 = plt.subplots()

                    im3 = ax3.imshow(sum_eigs_mat, vmin=abs(sum_eigs_mat).min(), vmax=abs(sum_eigs_mat).max(), cmap=PALETTE, origin='lower',
                                     extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
                    # im3 = ax3.imshow(sum_eigs_mat, vmin=0, vmax=0.3,
                    #                  cmap=plt.cm.CMRmap, origin='lower',
                    #                  extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
                    fig3.colorbar(im3, ax=ax3)
                    im3.set_clim(0., 2.)

                    ax3.set_title('Metric: 'r'$\sum_{i=1}^n{\| \lambda_i \|}$')

                    fname3 = os.path.join(outdir, f"sum_eig_region_{combo_string}.png")
                    plt.savefig(fname3)
                    plt.close('all')


if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 100
    nx = 2

    outdir = "./plots_all_20210106"
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
        ("softSVD", -1.5, -1.1, False),
        ("softSVD", 0.0, 1.0, False),
        ("softSVD", 0.99, 1.0, False),
        ("softSVD", 0.99, 1.1, False),
        ("softSVD", 1.0, 1.01, False),
        ("softSVD", 0.99, 1.01, False),
        ("softSVD", 1.1, 1.5, False),
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

    layers = [1, 2, 3, 4]

    # phase_and_spectra_plot_loop(10, layers, maps, activations)
    lin_regions(2, layers, maps, activations, outdir=outdir)
