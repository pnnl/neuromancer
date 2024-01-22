"""
Set of useful function transformations

"""

import torch

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


functions = {
    "bounds_scaling": bounds_scaling,
    "bounds_clamp": bounds_clamp,
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
