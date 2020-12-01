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

    fx(x) = sigma(Ax +b)
    lambda_ab*(A*x +b)

    """
    # TODO: there is problem with accurately extracting linear transform with bias
    #  Axb is not equal to Ax_b!
    Axb = fx.linear[0](x)
    lambda_Axb = fx.nonlin[0](Axb) / Axb
    A = fx.linear[0].effective_W()
    b = fx.linear[0].linear.bias
    Ax = torch.matmul(x, A)
    Ax_b = torch.matmul(x, A) + b
    lambda_h = fx.nonlin[0](Ax) / Ax
    lambda_h_b = fx.nonlin[0](Ax_b) / Ax_b
    lambda_b = fx.nonlin[0](b) / b
    print(f'{fx(x)} - network f(x)')
    print(f'{lambda_h * Ax} - A* without bias')
    print(f'{lambda_Axb * Axb} - A* with bias')
    print(f'{lambda_h_b * Ax_b} - A* with bias')
    print(f'{lambda_h_b * Ax + lambda_h_b*b} - A* with bias')
    # print(f'{lambda_h * Ax_b + lambda_b*b} - A* with bias')
    # print(f'{lambda_h * Ax + lambda_b*b} - A* with bias')


def LPV_net(fx, x):
    """
    verify that: f(x) = A_star*x
    f(x): multi-layer MLP
    A_star: parameter varying liner map
    x: input feature
    """
    # nonlinearities = fx.nonlin
    x_layer = x
    x_layer_b = x
    x_layer_orig = x
    x_layer_A_prime = x
    x_layer_A_prime_b = x
    W_weight = []
    W_activation = []
    W_layer = []
    i = 0
    for nlin, lin in zip(fx.nonlin, fx.linear):
        # original nonlinear map
        x_layer_orig = nlin(lin(x_layer_orig))

        ####################
            # BIAS NO
        ####################
        # layer-wise parameter vayring linear map no bias
        A = lin.effective_W()                       # layer weight
        Ax = torch.matmul(x_layer, A)               # linear transform
        if sum(Ax.squeeze()) == 0:
            lambda_h = torch.zeros(Ax.shape)
            # TODO: division by zero elementwise not replacing whole vector
            # TODO: replace zero values with derivatives of activations
        else:
            lambda_h = nlin(Ax)/Ax     # activation scaling
        lambda_h_matrix = torch.diag(lambda_h.squeeze())
        # x = lambda_h*Ax
        x_layer = torch.matmul(Ax, lambda_h_matrix)
        # layer-wise parameter vayring linear map: A' = Lambda A
        A_prime = torch.matmul(A, lambda_h_matrix)
        x_layer_A_prime = torch.matmul(x_layer_A_prime, A_prime)

        ####################
            # BIAS YES
        ####################
        # TODO: watch out, in lin there are two weights and biases
        #  for reconstruction we need to use lin.linear.weight.T and lin.linear.bias, not lin.weight and lin.bias
        # layer-wise parameter vayring linear map WITH bias
        A = lin.effective_W()                       # layer weight
        b = lin.linear.bias if  lin.linear.bias is not None else torch.zeros(x_layer_b.shape)
        Ax_b = torch.matmul(x_layer_b, A) + b  # affine transform
        # Ax_b = lin(x_layer_b)

        if sum(Ax_b.squeeze()) == 0:
            lambda_h_b = torch.zeros(Ax_b.shape)
        else:
            lambda_h_b = nlin(Ax_b)/Ax_b     # activation scaling
        lambda_h_matrix_b = torch.diag(lambda_h_b.squeeze())
        x_layer_b = torch.matmul(Ax_b, lambda_h_matrix_b)
        # layer-wise parameter vayring linear map: A' = Lambda A, x = A'x + Lambda b
        A_prime_b = torch.matmul(A, lambda_h_matrix_b)
        b_prime = lambda_h_b*b
        #  x = Lambda Ax + Lambda b   --> this holds because scaling is additive
        x_layer_A_prime_b = torch.matmul(x_layer_A_prime_b, A_prime_b) + b_prime

        # Prints
        print(f'layer {i+1}')
        print(f'{x_layer_orig} - original layer')
        print(f'{x_layer_b} - layer wise LPV with bias')
        print(f'{x_layer_A_prime_b} - layer wise LPV matrix with bias')
        print(f'{x_layer} - layer wise LPV no bias')
        print(f'{x_layer_A_prime} - layer wise LPV matrix no bias')

        # network-wise parameter vayring linear map:  A* = A'_L ... A'_1
        if i<1:
            A_star = A_prime
            A_star_b = A_prime_b
            b_star = b_prime
        else:
            # A* = A'A*
            A_star = torch.matmul(A_star, A_prime)
            A_star_b = torch.matmul(A_star_b, A_prime_b)
            b_star = torch.matmul(b_star, A_prime_b) + b_prime

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
    print(f'{fx(x)} - network f(x)')
    print(f'{torch.matmul(x, A_star_b)+b_star} - A* with bias')
    # print(f'{torch.matmul(x, A_star_b)} - A* with bias')
    print(f'{torch.matmul(x, A_star)} - A* without bias')
    return A_star, W_weight, W_activation, W_layer, w_net

nx = 3
# random feature point
x_z = torch.randn(1,nx)
# x_z = torch.tensor([[-1.0, 1.0, 3.0]], requires_grad=True)
test_bias = True

# define single layer square neural net
fx_layer = MLP_layer(nx, nx, nonlin=nn.ReLU, hsizes=[], bias=test_bias)
# verify linear operations on MLP layers
fx_layer.linear[0](x_z)
torch.matmul(x_z, fx_layer.linear[0].effective_W()) + fx_layer.linear[0].bias
# verify single layer linear parameter varying form
LPV_layer(fx_layer,torch.randn(1,3))

# define square neural net
fx = MLP(nx, nx, nonlin=nn.ReLU, hsizes=[nx, nx, nx], bias=test_bias)
if test_bias:
    for i in range(nx):
        fx.linear[i].bias.data = torch.randn(1,3)
# verify multi-layer linear parameter varying form
A_star, W_weight, W_activation, W_layer, w_net = LPV_net(fx,torch.randn(1,3))


# verify different activations
activations = [nn.ReLU6, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU,
              nn.LogSigmoid, nn.Sigmoid, nn.Tanh]
for act in activations:
    print(f'current activation {act}')
    fx_a = MLP(nx, nx, nonlin=act, hsizes=[nx, nx, nx], bias=test_bias)
    A_star, W_weight, W_activation, W_layer, w_net = LPV_net(fx_a, torch.randn(1, 3))

