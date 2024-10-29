"""
Set of useful function transformations

"""

import torch
import numpy as np

def bounds_scaling(x, xmin, xmax, scaling=1.):
    """
    hard bounds on variable x via sigmoid scaling between xmin and xmax values
    :param x:
    :param xmin:
    :param xmax:
    :param scaling:
    :return:
    """
    x = (xmax - xmin) * torch.sigmoid(scaling * x) + xmin
    return x


def bounds_clamp(x, xmin=None, xmax=None):
    """
    hard bounds on variable x via ReLU clamping between xmin and xmax values
    :param x:
    :param xmin:
    :param xmax:
    :return:
    """
    if xmin is not None:
        x = x + torch.relu(-x + xmin)
    if xmax is not None:
        x = x - torch.relu(x - xmax)
    return x


def window_functions(x, num_domains, delta=1.9):
    """
    Window functions for finite-basis domain decomposition.
    :param x: input tensor representing the domain.
    :param num_domains: number of domains. Must be a perfect square.
    :param delta: overlapping ratio. Higher = more overlapping.
    :return w/(s+eps): weighted window functions.
    """
    eps = 1e-12  # Small epsilon to prevent division by zero
    
    def w_jl_i(x_i, n_domains, x_min, x_max):
        jvec = torch.arange(n_domains, device=x_i.device) + 1
        muvec = x_min + (jvec - 1) * (x_max - x_min) / (n_domains - 1)
        muvec = muvec.unsqueeze(0).expand(x_i.shape[0], n_domains)
        u = x_i.repeat(1, n_domains)
        sigma = (x_max - x_min) * (delta / 2.0) / (n_domains - 1)

        z = (u - muvec) / (sigma + eps)
        w_jl = ((1 + torch.cos(np.pi * z)) / 2) ** 2
        w_jl = torch.where(torch.abs(z) < 1, w_jl, torch.zeros_like(w_jl))
        return w_jl

    n_dims = x.shape[1]
    if n_dims == 1:
        x_min, x_max = x.min(), x.max()
        w = w_jl_i(x, num_domains, x_min, x_max)
    elif n_dims == 2:
        n_per_dim = int(np.sqrt(num_domains))
        if n_per_dim ** 2 != num_domains:
            raise ValueError("num_domains must be a perfect square for 2D inputs.")
        x_min_x, x_max_x = x[:, 0].min(), x[:, 0].max()
        x_min_y, x_max_y = x[:, 1].min(), x[:, 1].max()
        w1 = w_jl_i(x[:, 0:1], n_per_dim, x_min_x, x_max_x)
        w2 = w_jl_i(x[:, 1:2], n_per_dim, x_min_y, x_max_y)  # Corrected slice
        w = torch.einsum('bi,bj->bij', w1, w2).reshape(x.shape[0], -1)
        if w.shape[1] > num_domains:
            w = w[:, :num_domains]  # Trim if necessary
    else:
        raise ValueError("Only 1D and 2D inputs are currently supported")

    s = torch.sum(w, dim=1, keepdim=True)
    return w / (s + eps)


functions = {
    "bounds_scaling": bounds_scaling,
    "bounds_clamp": bounds_clamp,
    "window_functions": window_functions,
}


if __name__ == "__main__":
    """
    examples for using bounds wrapped by Node class
    """

    from neuromancer.system import Node
    # wrapp bounds in Node class
    node_scale = Node(bounds_scaling, ['x', 'xmin', 'xmax'], ['x_new'], name='bounds_scaling')
    node_clamp = Node(bounds_clamp, ['x', 'xmin', 'xmax'], ['x_new'], name='bounds_clamp')
    # generate input data
    data = {'x': torch.randn(500, 5), 'xmin': -torch.ones(500, 5), 'xmax': torch.ones(500, 5)}
    # evaluate functions
    out_scale = node_scale(data)
    out_clamp = node_clamp(data)
    # check bounds satisfaction
    print(torch.all(out_scale['x_new'] <= data['xmax']))
    print(torch.all(out_scale['x_new'] >= data['xmin']))
    print(torch.all(out_clamp['x_new'] <= data['xmax']))
    print(torch.all(out_clamp['x_new'] >= data['xmin']))
