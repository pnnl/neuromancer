import sys
import os
from glob import glob

import torch
from torch import nn
import numpy as np
import slim
from neuromancer import blocks
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import dill

os.sys.path.append("neuromancer/train_scripts")
os.sys.path.append("neuromancer/train_scripts/stability_l4dc")

from lpv import lpv_batched
from phase_plots import *
from eigen_plots import compute_eigenvalues

PALETTE = sns.color_palette(
    "crest_r", as_cmap=True
)  # sns.color_palette("crest_r", as_cmap=True)

PALETTE.set_bad(PALETTE.colors[0])


class FxFy(nn.Module):
    def __init__(self, state_estimator, fx, fy):
        super().__init__()

        self.in_features = 2
        self.out_features = fy.out_features

        self.estimator = state_estimator

        self.linear = [*list(fx.linear), fy]
        for i, lin in enumerate(self.linear):
            if isinstance(lin, nn.Linear):
                w = lin.weight.clone()
                b = lin.bias.clone() if lin.bias is not None else None
                lin = slim.Linear(w.shape[1], w.shape[0], bias=b is not None)
                lin.linear.weight = nn.Parameter(w)
                lin.linear.bias = nn.Parameter(b) if b is not None else None
                self.linear[i] = lin
        self.nonlin = [*fx.nonlin, nn.Identity()]
        self.nhidden = fx.nhidden + 2

    def initial_state(self, x):
        x = {"Yp": x.expand(self.estimator.nsteps, -1, -1)}
        x = self.estimator(x)
        return x

    def forward(self, x):
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x


class FuFy(nn.Module):
    def __init__(self, fu, fy):
        super().__init__()

        self.in_features = fu.in_features
        self.out_features = fy.out_features

        self.linear = [*list(fu.linear), fy]
        for i, lin in enumerate(self.linear):
            if isinstance(lin, nn.Linear):
                w = lin.weight.clone()
                b = lin.bias.clone() if lin.bias is not None else None
                lin = slim.Linear(w.shape[1], w.shape[0], bias=b is not None)
                lin.linear.weight = nn.Parameter(w)
                lin.linear.bias = nn.Parameter(b) if b is not None else None
                self.linear[i] = lin
        self.nonlin = [*fu.nonlin, nn.Identity()]
        self.nhidden = fu.nhidden + 1

    def forward(self, x):
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x


def compute_norms(matrices):
    m_norms = []
    for m in matrices:
        assert len(m.shape) == 2
        m_norm= np.linalg.norm(m.T)
        m_norms += [m_norm]
    return m_norms


def mk_1d_region_plots(fx, x, name):
    x = x.flatten().unsqueeze(-1)
    Astars, Astar_b, *_ = lpv_batched(fx, x)
    Astars = Astars.detach().numpy()

    # plot Anorms
    Anorms = compute_norms(Astars)
    Anorm_mat = np.reshape(Anorms, x.shape)

    fig1, ax1 = plt.subplots()
    ax1.plot(x, Anorm_mat)
    fname1 = os.path.join(outdir, f"norm_region_{name}.pdf")
    plt.savefig(fname1)

    # plot dominant eigenvalues
    eigvals = compute_eigenvalues(Astars)
    dom_eigs = [np.absolute(eigs).max() for eigs in eigvals]
    dom_eigs_mat = np.reshape(dom_eigs, x.shape)

    fig2, ax2 = plt.subplots()

    vmin = abs(dom_eigs_mat).min()
    vmax = abs(dom_eigs_mat).max()

    ax2.plot(x, dom_eigs_mat)

    #ax2.set_title('Metric: 'r'$\| \lambda_1 \|$')

    fname2 = os.path.join(outdir, f"dom_eig_region_{name}.pdf")
    plt.savefig(fname2)

    # plot sum of absolute eigenvalues
    sum_eigs = [np.absolute(eigs).sum() for eigs in eigvals]
    sum_eigs_mat = np.reshape(sum_eigs, x.shape)

    fig3, ax3 = plt.subplots()
    ax3.plot(x, sum_eigs_mat)

    # im3 = ax3.imshow(sum_eigs_mat, vmin=0, vmax=0.3,
    #                  cmap=plt.cm.CMRmap, origin='lower',
    #                  extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
    #im3.set_clim(0., 2.)

    # ax3.set_title('Metric: 'r'$\sum_{i=1}^n{\| \lambda_i \|}$')

    fname3 = os.path.join(outdir, f"sum_eig_region_{name}.pdf")
    plt.savefig(fname3)
    plt.close('all')


def mk_region_plots(fx, input_grid, name, use_bias=False, initial_states=None):
    grid_x, grid_y = input_grid

    _, ax = plt.subplots()

    # TODO: skipping phase portrait plotting for models
    if isinstance(fx, FxFy):
        g = torch.stack((grid_x.flatten(), grid_y.flatten())).T
        g = fx.initial_state(g)["x0_estim"]
        Astars, Astar_b, *_ = lpv_batched(fx, g, use_bias=use_bias)
    else:
        X, Y, U, V, Astars = compute_grid(fx, input_grid, use_bias)

        ax = plot_model_streamlines(
            X, Y, U, V,
            x_lims=(-6, 6),
            y_lims=(-6, 6),
            step=0.5,
            ax=ax,
        )
        if len(initial_states) > 0:
            ax = plot_trajectories(fx, initial_states, use_bias=use_bias, ax=ax)

        plt.savefig(os.path.join(outdir, f"phase_{name}.pdf"))
        plt.close()

    X, Y = input_grid
    Astars = Astars.detach().numpy()

    #X = torch.stack((grid_x.flatten(), grid_y.flatten())).T
    #Astars, Astar_b, *_ = lpv_batched(fx, X, use_bias=use_bias)
    #Astars = Astars.detach().numpy()

    # plot Anorms
    Anorms = compute_norms(Astars)
    Anorm_mat = np.reshape(Anorms, grid_x.shape)

    fig1, ax1 = plt.subplots()

    im1 = ax1.imshow(Anorm_mat, #vmin=abs(Anorm_mat).min(), vmax=abs(Anorm_mat).max(),
               cmap=PALETTE, origin='lower', # norm=colors.LogNorm(),
               extent=[X.min(), X.max(), X.min(), X.max()])#, interpolation="bilinear")
    fig1.colorbar(im1, ax=ax1)
    #im1.set_clim(0., 2.)

    # ax1.set_title('Metric: 'r'$\Vert A^* \Vert$')

    fname1 = os.path.join(outdir, f"norm_region_{name}.pdf")
    plt.savefig(fname1)

    # plot dominant eigenvalues
    eigvals = compute_eigenvalues(Astars)
    dom_eigs = [np.absolute(eigs).max() for eigs in eigvals]
    dom_eigs_mat = np.reshape(dom_eigs, grid_x.shape)

    fig2, ax2 = plt.subplots()

    vmin = abs(dom_eigs_mat).min()
    vmax = abs(dom_eigs_mat).max()

    im2 = ax2.imshow(dom_eigs_mat,
                     cmap=PALETTE, origin='lower', # norm=colors.LogNorm(),
                     extent=[X.min(), X.max(), X.min(), X.max()])#, interpolation="bilinear")
    fig2.colorbar(im2, ax=ax2)

    #ax2.set_title('Metric: 'r'$\| \lambda_1 \|$')

    fname2 = os.path.join(outdir, f"dom_eig_region_{name}.pdf")
    plt.savefig(fname2)

    # plot sum of absolute eigenvalues
    sum_eigs = [np.absolute(eigs).sum() for eigs in eigvals]
    sum_eigs_mat = np.reshape(sum_eigs, grid_x.shape)

    fig3, ax3 = plt.subplots()

    im3 = ax3.imshow(sum_eigs_mat, #vmin=abs(sum_eigs_mat).min(), vmax=abs(sum_eigs_mat).max(),
                     cmap=PALETTE, origin='lower', # norm=colors.LogNorm(),
                     extent=[X.min(), X.max(), X.min(), X.max()])#, interpolation="bilinear")
    # im3 = ax3.imshow(sum_eigs_mat, vmin=0, vmax=0.3,
    #                  cmap=plt.cm.CMRmap, origin='lower',
    #                  extent=[X.min(), X.max(), X.min(), X.max()], interpolation="bilinear")
    fig3.colorbar(im3, ax=ax3)
    #im3.set_clim(0., 2.)

    # ax3.set_title('Metric: 'r'$\sum_{i=1}^n{\| \lambda_i \|}$')

    fname3 = os.path.join(outdir, f"sum_eig_region_{name}.pdf")
    plt.savefig(fname3)
    plt.close('all')


def lin_regions(input_grid, nx, layers, maps, activations, outdir="./plots_region", initial_states=None):
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
                    try:
                        mk_region_plots(fx, input_grid, combo_string, bias, initial_states)
                    except Exception as e:
                        print(e)


if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 100
    nx = 2

    outdir = "./plots_batched_20210527"
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
        #("gershgorin", -1.5, -1.1, True),
        ("gershgorin", 0.0, 1.0, True),
        ("gershgorin", 0.99, 1.0, True),
        #("gershgorin", 0.99, 1.1, True),
        #("gershgorin", 1.0, 1.01, True),
        #("gershgorin", 0.99, 1.01, True),
        #("gershgorin", 1.1, 1.5, True),
        #("gershgorin", -1.5, -1.1, False),
        ("gershgorin", 0.0, 1.0, False),
        #("gershgorin", 0.99, 1.0, False),
        #("gershgorin", 0.99, 1.1, False),
        #("gershgorin", 1.0, 1.01, False),
        #("gershgorin", 0.99, 1.01, False),
        #("gershgorin", 1.1, 1.5, False),
        #("softSVD", -1.5, -1.1, False),
        ("softSVD", 0.0, 1.0, False),
        #("softSVD", 0.99, 1.0, False),
        ("softSVD", 0.99, 1.1, False),
        #("softSVD", 1.0, 1.01, False),
        #("softSVD", 0.99, 1.01, False),
        #("softSVD", 1.1, 1.5, False),
        ("pf", 1.0, 1.0, False),
        #("linear", 1.0, 1.0, False),
    ]
    activations = [
        #("softplus", torch.nn.Softplus),
        #("sigmoid", torch.nn.Sigmoid),
        ("selu", torch.nn.SELU),
        #("tanh", torch.nn.Tanh),
        ("relu", torch.nn.ReLU),
    ]

    layers = [1, 4, 8]

    input_grid = torch.meshgrid(
        torch.arange(-6, 6.1, 0.1),
        torch.arange(-6, 6.1, 0.1),
    )

    torch.set_grad_enabled(False)

    for fname in glob("/qfs/projects/deepmpc/ape/stability_models_neurips/*.pt"):
        if "-blu" in fname or "-softexp" in fname or "_state_dict" in fname or "blocknlin-linear-linear-mlp-gelu" not in fname: continue
        #torch.manual_seed(408)
        #np.random.seed(408)
        name = fname.split("/")[-1].split(".")[0]
        print(name)
        model = torch.load(fname, pickle_module=dill, map_location="cpu")
        sd = torch.load(fname.replace(".pt", "_state_dict.pt"), map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        estim = model.components[0]
        fx = model.components[1].fx
        fy = model.components[1].fy
        fxfy = FxFy(estim, fx, fy)
        mk_region_plots(fxfy, input_grid, f"{name}_fx", use_bias=True, initial_states=initial_states)
        fu = model.components[1].fu
        fufy = FuFy(fu, fy)
        try:
            mk_region_plots(fufy, input_grid, f"{name}_fu", use_bias=True, initial_states=[])
        except Exception as e:
            mk_1d_region_plots(fufy, input_grid[0], f"{name}_fu")

    # phase_and_spectra_plot_loop(10, layers, maps, activations)
    #lin_regions(input_grid, 2, layers, maps, activations, outdir="./plots_region", initial_states=initial_states)
