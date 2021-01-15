import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../stability_l4dc")

from lpv import lpv, lpv_batched

def compute_P(model, x, t=100, use_bias=True):
    Astar_P = torch.eye(x.shape[0])
    P_sum = torch.zeros_like(Astar_P)
    Q = torch.eye(x.shape[0])
    for i in range(t):
        Astar, Astar_b, bstar, *_ = lpv(model, x)
        Astar_P = torch.matmul(Astar_P, Astar)
        P_sum = P_sum + torch.chain_matmul(Astar_P.T, Q, Astar_P)

        if use_bias:
            x = torch.matmul(x, Astar_b) + bstar
        else:
            x = torch.matmul(x, Astar)

    return P_sum


def plot(model, t=100, limits=(-6, 6), step=0.1):
    X, Y = torch.meshgrid(
        torch.arange(limits[0], limits[1] + step, step),
        torch.arange(limits[0], limits[1] + step, step),
    )
    grid = torch.stack((X.flatten(), Y.flatten())).T

    V_star = []
    P_sums = []
    A_stars = []
    for x in grid:
        P_sum = compute_P(model, x, t)
        P_sums.append(P_sum.detach().numpy())
        V_star.append(
            torch.matmul(torch.matmul(x.T, P_sum), x)
        )
        Astar, Astar_b, bstar, *_ = lpv(model, x)
        A_stars.append(Astar.detach().numpy())

    V_star_grid = torch.stack(V_star)
    V_star_grid = V_star_grid.T.reshape(*X.shape).detach().cpu().numpy()

    is_symmetric = np.array(list(map(lambda x: np.all(x == x.T), P_sums)))
    P_eigvals = np.array([np.linalg.eigvals(x) for x in P_sums])

    A_stars = np.stack(A_stars)
    Astar_std = np.std(A_stars, axis=(1, 2))
    Astar_std = Astar_std.T.reshape(*X.shape)

    print(P_eigvals)
    print(is_symmetric)

    is_pos_def = np.array(list(map(lambda x: np.all(x == x.T) and np.all(np.linalg.eigvals(x) >= -1e-6), P_sums)))
    print(is_pos_def)
    is_pos_def = is_pos_def.T.reshape(*X.shape)

    extent=[
        limits[0] - step / 2,
        limits[1] + step / 2,
        limits[0] - step / 2,
        limits[1] + step / 2,
    ]

    fig, ax = plt.subplots(ncols=4)
    im1 = ax[0].imshow(
        V_star_grid,
        extent=extent,
        # cmap=PALETTE,
    )
    ax[1].imshow(
        is_pos_def,
        extent=extent,
        # cmap=PALETTE,
    )
    ax[2].imshow(
        V_star_grid * is_pos_def,
        extent=extent,
        # cmap=PALETTE,
    )
    ax[3].imshow(
        Astar_std,
        extent=extent
    )
    fig.colorbar(im1, ax=ax)
    return [ax]

if __name__ == "__main__":
    import slim
    from neuromancer import blocks

    linmap = slim.linear.maps["gershgorin"]
    fx = blocks.MLP(
        2,
        2,
        bias=False,
        linear_map=linmap,
        nonlin=torch.nn.ReLU,
        hsizes=[2] * 2,
        linargs=dict(sigma_min=1.1, sigma_max=1.2, real=False),
    )

    plot(fx, t=10, step=1.)
    plt.show()