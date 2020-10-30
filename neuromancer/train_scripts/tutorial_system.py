from neuromancer.blocks import MLP
import torch.nn as nn
import torch
import slim
import scipy.linalg as LA
import numpy as np


class JacobianMLP(nn.Module):
    def __init__(self,  insize, outsize, bias=False,
                 Linear=slim.Linear, nonlin=nn.ReLU, hsizes=[64], linargs=dict()):

        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.linear = nn.ModuleList([Linear(sizes[k],
                                            sizes[k+1],
                                            bias=bias,
                                            **linargs) for k in range(self.nhidden+1)])

    def get_jacobian(self, x, xprev):
        grads = []
        for i in range(x.shape[-1]):
            grads.append(torch.autograd.grad(x.flatten()[i], xprev, retain_graph=True)[0].flatten())
        return torch.stack(grads).T

    def forward(self, x):
        """
        Data Jacobian constructed from equation (9)
        from https://people.ece.uw.edu/bilmes/p/mypubs/wang-dnn-jacobian-2016.pdf
        :param x:
        :return:
        """
        jacobians = []
        xprev = x
        for lin, nlin in zip(self.linear, self.nonlin):
            x = lin(xprev)
            jacobians.append(self.get_jacobian(x, xprev))
            xprev = x
            x = nlin(xprev)
            jacobians.append(self.get_jacobian(x, xprev))
            xprev = x
        DJM = torch.chain_matmul(*jacobians)
        return x, DJM, jacobians


# Faster to calculate the DJM this way
x = torch.tensor([[-1.0, 1.0, 3.0, 5.0]], requires_grad=True)
fx = MLP(4, 4, nonlin=nn.ReLU, hsizes=[4, 64, 128, 4], bias=False)
grads = []
for i in range(4):
    grads.append(torch.autograd.grad(fx(x).flatten()[i], x)[0].flatten())
DJM = torch.stack(grads).T
print(fx(x))
print(torch.matmul(x, DJM))

# Sanity check to view the factorized DJM
fxj = JacobianMLP(4, 4, nonlin=nn.ReLU, hsizes=[4, 64, 28, 4])
output, DJM, jacobians = fxj(x)
print(output)
print(torch.matmul(x, DJM))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #  Deep neural nets are linear parameter varying maps   # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class MLP_layer(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=nn.ReLU, hsizes=[64], linargs=dict()):
        """

        :param insize:
        :param outsize:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param linargs: (dict) arguments for
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin() for k in range(self.nhidden)] + [nonlin()]
        self.linear = nn.ModuleList([Linear(sizes[k],
                                            sizes[k+1],
                                            bias=bias,
                                            **linargs) for k in range(self.nhidden+1)])

    def reg_error(self):
        """

        :return:
        """
        return sum([k.reg_error()
                    for k in self.linear
                    if hasattr(k, 'reg_error')])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x


def LPV_layer(fx,x):
    """
    verify that: f(x) = lambda_h*A*x
    f(x): single layer MLP
    lambda_h: parameter varying scaling diagonal matrix of activations
    A: layer weight
    x: input feature
    """
    print(fx(x))
    fx.linear[0](x)
    A = fx.linear[0].effective_W()
    Ax = torch.matmul(x, A)
    lambda_h = fx.nonlin[0](Ax) / Ax
    print(lambda_h * Ax)


def LPV_net(fx, x):
    # nonlinearities = fx.nonlin
    x_layer = x
    x_layer_orig = x
    x_layer_A_prime = x
    W_weight = []
    W_activation = []
    W_layer = []
    i = 0
    for nlin, lin in zip(fx.nonlin, fx.linear):
        # layer-wise parameter vayring linear map
        A = lin.effective_W()                     # layer weight
        Ax = torch.matmul(x_layer, A)           # linear transform
        lambda_h = nlin(Ax)/Ax     # activation scaling # TODO: treat division by zero
        lambda_h_matrix = torch.diag(lambda_h.squeeze())
        # x = lambda_h*Ax
        x_layer = torch.matmul(Ax, lambda_h_matrix)
        x_layer_orig = nlin(lin(x_layer_orig))

        # layer-wise parameter vayring linear map: A' = Lambda A
        A_prime = torch.matmul(A, lambda_h_matrix)
        x_layer_A_prime = torch.matmul(x_layer_A_prime, A_prime)
        print(f'layer {i+1}')
        print(x_layer_orig)
        print(x_layer)
        print(x_layer_A_prime)

        # network-wise parameter vayring linear map:  A* = A'_L ... A'_1
        if i<1:
            A_star = A_prime
        else:
            # A* = A'A*
            A_star = torch.matmul(A_star, A_prime)
        i+=1
        # layer eigenvalues
        w_weight, v = LA.eig(lin.weight.detach().cpu().numpy().T)
        w_activation, v = LA.eig(lambda_h_matrix.detach().cpu().numpy().T)
        w_layer, v = LA.eig(A_prime.detach().cpu().numpy().T)
        W_weight.append(w_weight)
        W_activation.append(w_activation)
        W_layer.append(w_layer)
        # print(f'eigenvalues of {i}-th layer weights {w_weight}')
        # print(f'eigenvalues of {i}-th layer activations {w_activation}')
    # network-wise eigenvalues
    w_net, v = LA.eig(A_prime.detach().cpu().numpy().T)
    print(f'point-wise eigenvalues of network {w_net}')
    print(f'network forward pass vs LPV')
    print(fx(x))
    print(torch.matmul(x, A_star))
    return A_star, W_weight, W_activation, W_layer, w_net

nx = 3
# random feature point
x_z = torch.randn(1,nx)
# x_z = torch.tensor([[-1.0, 1.0, 3.0]], requires_grad=True)

# define square neural net
fx = MLP(nx, nx, nonlin=nn.ReLU, hsizes=[nx, nx, nx], bias=False)
# define single layer square neural net
fx_layer = MLP_layer(nx, nx, nonlin=nn.ReLU, hsizes=[], bias=False)

# verify linear operations on MLP layers
fx_layer.linear[0](x_z)
torch.matmul(x_z, fx_layer.linear[0].effective_W()) + fx_layer.linear[0].bias


# verify single layer linear parameter varying form
LPV_layer(fx_layer,torch.randn(1,3))
# verify multi-layer linear parameter varying form
A_star, W_weight, W_activation, W_layer, w_net = LPV_net(fx,torch.randn(1,3))
