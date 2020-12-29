import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../stability_l4dc")
)

import torch
import numpy as np
from lpv import lpv_batched, lpv
from eigen_plots import compute_eigenvalues, plot_eigenvalues
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_eigenvalues_set(fx, nx, limits=(-6, 6), nsamples=1000, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)
    Astars = Astars.detach().numpy()
    eigvals = compute_eigenvalues(Astars)
    plot_eigenvalues(eigvals, fname=fname)


def plot_phase_portrait_hidim(fx, nx, limits=(-6, 6), nsamples=1000, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)

    XY = torch.empty(nsamples, 2)
    UV = torch.empty(nsamples, 2)
    for i, (Astar, x_sample) in enumerate(zip(Astars, x)):
        z_sample = torch.matmul(Astar, x_sample)
        u, s, vh = torch.svd(Astar)
        # vh = u[:2, ...]
        # # TODO Jan: I believe we want to use right singular vectors
        # #  for projection onto principal components
        vh = vh[:2, ...]
        XY[i, ...] = torch.matmul(vh, x_sample)
        UV[i, ...] = torch.matmul(vh, z_sample)

    X, Y = XY.T.detach().numpy()
    U, V = UV.T.detach().numpy()

    # plot vector field
    fig, ax = plt.subplots()
    ax.quiver(
        X, Y, U, V, angles="uv", pivot="mid", width=0.002, headwidth=4, headlength=5
    )
    ax.set_aspect(1)

    if fname is not None:
        plt.savefig(fname)
        plt.close()

    return [ax]


def plot_singular_values(fx, nx, limits=(-6, 6), nsamples=10, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)

    fig, ax = plt.subplots()
    for i, Astar in enumerate(Astars):
        _, s, _ = torch.svd(Astar, compute_uv=False)
        plt.plot(np.arange(1, nx + 1, 1), s.T.detach().numpy())
        plt.grid(True)
        plt.xlabel('$k$')
        plt.ylabel('$\sigma_k$')
    ax.set_yscale("log")

    if fname is not None:
        plt.savefig(fname)
        plt.close()

    return [ax]


def plot_Jacobian_norms(fx, nx, limits=(-6, 6), nsamples=10, fname=None):
    x = torch.rand(nsamples, nx, dtype=torch.float) * (limits[1] - limits[0]) + limits[0]
    Astars, *_ = lpv_batched(fx, x)
    Astars_np = Astars.detach().numpy()

    Astars_mean = np.mean(Astars_np, 0)
    Astars_var = np.var(Astars_np, 0)
    Astars_min = np.min(Astars_np, 0)
    Astars_max = np.max(Astars_np, 0)

    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(Astars_mean, vmin=abs(Astars_mean).min(), vmax=abs(Astars_mean).max(), cmap=plt.cm.CMRmap)
    ax[0].set_title('mean($A^{\star}}$)')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)
    im2 = ax[1].imshow(Astars_var, vmin=abs(Astars_var).min(), vmax=abs(Astars_var).max(), cmap=plt.cm.CMRmap)
    ax[1].set_title('var($A^{\star}}$)')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)
    # im3 = ax[1,0].imshow(Astars_min, vmin=abs(Astars_min).min(), vmax=abs(Astars_min).max(), cmap=plt.cm.CMRmap)
    # ax[1, 0].set_title('min($A^{\star}}$)')
    # fig.colorbar(im3, ax=ax[1,0])
    # im4 = ax[1,1].imshow(Astars_max, vmin=abs(Astars_max).min(), vmax=abs(Astars_max).max(), cmap=plt.cm.CMRmap)
    # ax[1, 1].set_title('max($A^{\star}}$)')
    # fig.colorbar(im4, ax=ax[1,1])
    fig.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        plt.close()

if __name__ == "__main__":
    # TODO(lltt): plot singular values
    import slim
    from neuromancer import blocks

    outdir = "plots_20201228_paper"
    os.makedirs(outdir, exist_ok=True)
    SEED = 410
    nx = 64

    linmap = slim.linear.maps["gershgorin"]
    nonlin = torch.nn.SELU
    fx = blocks.MLP(
        nx,
        nx,
        bias=False,
        linear_map=linmap,
        nonlin=nonlin,
        hsizes=[nx] * 4,
        linargs=dict(sigma_min=0.5, sigma_max=1.2, real=False),
    )

    plot_singular_values(fx, nx)
    plt.show()
    plot_phase_portrait_hidim(fx, nx, limits=(-6, 6))
    plot_Jacobian_norms(fx, nx, limits=(-6, 6))


    linmaps = ['spectral', "damp_skew_symmetric", "skew_symetric", "symplectic",
               'linear', "gershgorin", "pf", "softSVD"]
    # sigma_min_max = [(0.0, 0.5),  (0.5, 1.0), (0.8, 1.2), (-1.5, -1.1)]
    sigma_min_max = [(0.9, 1.0)]

    # sigma_min_max = [(0.0, 0.5),  (0.0, 1.0), (0.5, 1.0), (0.99, 1.0),
    #                 (1.0, 1.0),  (0.99, 1.1), (0.8, 1.2), (1.0, 1.5),
    #                 (-1.5, -1.1), (-1.0, -0.5), (-1.0, 1.0), (-2.0, 2.0)]

    activations = {"relu": torch.nn.ReLU}

    # activations = {
    #     "relu": torch.nn.ReLU,
    #     "selu": torch.nn.SELU,
    #     "gelu": torch.nn.GELU,
    #     "tanh": torch.nn.Tanh,
    #     "sigmoid": torch.nn.Sigmoid,
    #     "softplus": torch.nn.Softplus,
    #     "identity": torch.nn.Identity
    # }

    combos = [
        (("softSVD", 0.5, 1.0, False), 8, [("identity", False)]),
        (("softSVD", 0.8, 1.2, False), 8, [("identity", False)]),
        (("gershgorin", 0.0, 1.0, True), 1, [("identity", False)]),
        (("gershgorin", 0.0, 1.0, True), 8, [("identity", False)]),
        (("gershgorin", 0.8, 1.2, True), 8, [("identity", False)]),
        (("gershgorin", 0.0, 1.0, False), 1, [("identity", False)]),
        (("gershgorin", 0.0, 1.0, False), 8, [("identity", False)]),
        (("gershgorin", 0.8, 1.2, False), 8, [("identity", False)]),
        (("pf", 0.5, 1.0, False), 8, [("identity", False)]),
        (("pf", 0.8, 1.2, False), 8, [("identity", False)]),

        (("softSVD", 0.5, 1.0, False), 8, [("selu", False)]),
        (("softSVD", 0.5, 1.0, False), 8, [("tanh", False)]),
        (("softSVD", 0.5, 1.0, False), 8, [("relu", False)]),
        (("softSVD", 0.5, 1.0, False), 8, [("softplus", False)]),
        (("softSVD", 0.99, 1.0, False), 8, [("selu", False)]),
        (("softSVD", 0.99, 1.01, False), 8, [("selu", False)]),
        (("softSVD", 0.99, 1.1, False), 8, [("selu", False)]),
        (("softSVD", 1.0, 1.01, False), 8, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 8, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 8, [("tanh", False)]),
        (("softSVD", 0.99, 1.1, False), 8, [("tanh", False)]),
        (("softSVD", 0.5, 1.0, False), 1, [("relu", False)]),
        (("softSVD", 0.5, 1.0, False), 1, [("softplus", False)]),
        (("softSVD", 0.5, 1.0, False), 1, [("selu", False)]),
        (("softSVD", 0.5, 1.0, False), 1, [("tanh", False)]),
        (("softSVD", 0.99, 1.0, False), 1, [("selu", False)]),
        (("softSVD", 0.99, 1.01, False), 1, [("selu", False)]),
        (("softSVD", 0.99, 1.1, False), 1, [("selu", False)]),
        (("softSVD", 1.0, 1.01, False), 1, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 1, [("selu", False)]),
        (("softSVD", -1.5, -1.1, False), 1, [("tanh", False)]),
        (("softSVD", 0.99, 1.1, False), 1, [("tanh", False)]),

        (("gershgorin", 0.0, 1.0, False), 8, [("relu", False)]),
        (("gershgorin", 0.0, 1.0, False), 8, [("tanh", False)]),
        (("gershgorin", 0.0, 1.0, False), 8, [("gelu", False)]),
        (("gershgorin", 0.0, 1.0, False), 8, [("selu", False)]),
        (("gershgorin", -1.5, -1.1, False), 8, [("selu", True)]),
        (("gershgorin", -1.5, -1.1, False), 8, [("gelu", True)]),
        (("gershgorin", -1.5, -1.1, False), 8, [("relu", True)]),
        (("gershgorin", -1.5, -1.1, False), 8, [("softplus", True)]),
        (("gershgorin", 0.99, 1.0, False), 8, [("gelu", False)]),
        (("gershgorin", 0.99, 1.0, False), 8, [("softplus", False)]),
        (("gershgorin", 0.99, 1.0, False), 8, [("tanh", False)]),
        (("gershgorin", 0.99, 1.1, False), 8, [("softplus", False)]),
        (("gershgorin", 0.99, 1.1, False), 8, [("gelu", True)]),
        (("gershgorin", 0.99, 1.1, False), 8, [("tanh", True)]),
        (("gershgorin", 1.0, 1.01, False), 8, [("softplus", True)]),
        (("gershgorin", 1.0, 1.01, False), 8, [("tanh", True)]),
        (("gershgorin", 1.0, 1.01, False), 8, [("relu", True)]),

        (("pf", 1.0, 1.0, False), 8, [("relu", False)]),
        (("pf", 0.5, 1.0, False), 8, [("relu", False)]),
        (("pf", 1.0, 1.0, False), 8, [("softplus", False)]),
        (("pf", 0.5, 1.0, False), 8, [("softplus", False)]),
        (("pf", 1.0, 1.0, False), 8, [("gelu", False)]),
        (("pf", 0.5, 1.0, False), 8, [("gelu", False)]),
        (("pf", 1.0, 1.0, False), 8, [("tanh", False)]),
        (("pf", 0.5, 1.0, False), 8, [("tanh", False)]),
        (("pf", 1.0, 1.0, False), 8, [("sigmoid", False)]),
        (("pf", 0.5, 1.0, False), 8, [("sigmoid", False)]),
        (("pf", 1.0, 1.0, False), 1, [("relu", False)]),
        (("pf", 0.5, 1.0, False), 1, [("relu", False)]),
        (("pf", 1.0, 1.0, False), 1, [("softplus", False)]),
        (("pf", 0.5, 1.0, False), 1, [("softplus", False)]),
        (("pf", 1.0, 1.0, False), 1, [("gelu", False)]),
        (("pf", 0.5, 1.0, False), 1, [("gelu", False)]),
        (("pf", 1.0, 1.0, False), 1, [("tanh", False)]),
        (("pf", 0.5, 1.0, False), 1, [("tanh", False)]),
        (("pf", 1.0, 1.0, False), 1, [("sigmoid", False)]),
        (("pf", 0.5, 1.0, False), 1, [("sigmoid", False)]),
    ]

    # for (linmap, sigmin, sigmax, real), nlayers, params in combos:
    #     for act_name, bias in params:
    nlayers = 4
    real = False
    bias = False
    for linmap in linmaps:
        for (sigmin, sigmax) in sigma_min_max:
            for act_name in activations.keys():
                combo_string = f"{linmap}_x{nx}_({sigmin},{sigmax})_{act_name}_{nlayers}l_{'real' if real else 'complex'}"
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                fx = blocks.MLP(
                    nx,
                    nx,
                    nonlin=activations[act_name],
                    linear_map=slim.linear.maps[linmap],
                    hsizes=[nx] * nlayers,
                    bias=True,
                    linargs={
                        "sigma_min": sigmin,
                        "sigma_max": sigmax,
                        "real": real,
                    },
                )

                plot_eigenvalues_set(fx, nx, nsamples=1000,  fname=os.path.join(outdir, f"Eig_values_{combo_string}.png"))
                plot_singular_values(fx, nx, nsamples=1000, fname=os.path.join(outdir, f"S_values_{combo_string}.png"))
                plot_phase_portrait_hidim(fx, nx, limits=(-6, 6), nsamples=1000, fname=os.path.join(outdir, f"phase_plot_{combo_string}.png"))
                plot_Jacobian_norms(fx, nx, limits=(-6, 6), nsamples=1000, fname=os.path.join(outdir, f"Jacob_plot_{combo_string}.png"))
