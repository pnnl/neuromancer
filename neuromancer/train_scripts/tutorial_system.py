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


# eigenvalues of square neural nets
x = torch.tensor([[-1.0, 1.0, 3.0]], requires_grad=True)
fx = MLP(3, 3, nonlin=nn.ReLU, hsizes=[3, 3, 3], bias=False)

nonlinearities = fx.nonlin
W = []
for i, m in enumerate(fx.linear):
    # layer-wise parameter vayring linear map
    Ax = torch.matmul(x, m.weight)
    lambda_h = nonlinearities[i](Ax)/Ax
    lambda_h_matrix = torch.diag(lambda_h.squeeze())
    # x = lambda_h*Ax
    x = torch.matmul(Ax, lambda_h_matrix)

    # network-wise parameter vayring linear map
    A_prime_h = torch.matmul(m.weight, lambda_h_matrix)
    if i<1:
        A_prime = A_prime_h
    else:
        A_prime = torch.matmul(A_prime, A_prime_h)

    # layer eigenvalues
    w_weight, v = LA.eig(m.weight.detach().cpu().numpy().T)
    w_activation, v = LA.eig(lambda_h_matrix.detach().cpu().numpy().T)
    print(f'eigenvalues of {i}-th layer weights {w_weight}')
    print(f'eigenvalues of {i}-th layer activations {w_activation}')

    # network-wise eigenvalues
    w_net, v = LA.eig(A_prime.detach().cpu().numpy().T)
    print(f'point-wise eigenvalues of network {w_net}')