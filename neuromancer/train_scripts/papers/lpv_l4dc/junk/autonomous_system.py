from neuromancer.blocks import MLP
import torch.nn as nn
import torch
import slim
import scipy.linalg as LA


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
        if sum(Ax.squeeze()) == 0:
            lambda_h = torch.zeros(Ax.shape)
        else:
            lambda_h = nlin(Ax)/Ax     # activation scaling
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


class AutonomousSystem(nn.Module):

    def __init__(self, nx, hsizes, act, linearmap, sigmin, sigmax, real=True, bias=False):
        """

        :param nx: state dimension
        :param hsizes: list of hidden state sizes (don't need to be the same as state dimension)
        :param act: callable elemntwise nonlinear function that operates on pytorch tensors
        :param linearmap: class which inherits from slim.linear.LinearBase
        :param sigmin: lower bound on eigenvalues
        :param sigmax: upper bound on eigenvalues
        """
        super().__init__()
        self.fx = MLP(nx, nx, nonlin=act, Linear=linearmap, hsizes=hsizes, bias=bias, linargs={'sigma_min': sigmin,
                                                                                               'sigma_max': sigmax,
                                                                                               'real': real})

    def forward(self, x, nsim=1):
        As, wnets, wweights, wacts, wlayers = [], [], [], [], []
        for i in range(nsim):
            A_star, W_weight, W_activation, W_layer, w_net = LPV_net(self.fx, x)
            As.append(A_star)
            wnets.append(w_net)
            wweights.append(W_weight)
            wacts.append(W_activation)
            wlayers.append(W_layer)
            x = self.fx(x)
        return x, As, wnets, wweights, wacts, wlayers


square_maps = [(slim.linear.maps['gershgorin'], -1.5, -1.1),
               (slim.linear.maps['gershgorin'], 0.0, 1.0),
               (slim.linear.maps['gershgorin'], .99, 1.1),
               (slim.linear.maps['gershgorin'], 1.1, 1.5),
               (slim.linear.maps['pf'], 1.0, 1.0),
               (slim.linear.maps['linear'], 1.0, 1.0)]

maps = [(slim.linear.maps['softSVD'], -1.5, -1.1),
        (slim.linear.maps['softSVD'], 0.0, 1.0),
        (slim.linear.maps['softSVD'], .99, 1.1),
        (slim.linear.maps['softSVD'], 1.1, 1.5),
        (slim.linear.maps['softSVD'], 1.0, 1.0),
        (slim.linear.maps['linear'], 1.0, 1.0)]

if __name__ == '__main__':
    nx = 2
    # for nlayers in [1, 8]:
    #     for hsize in [2]:
    #         for linmap, sigmin, sigmax in maps:
    #             for act in [nn.RelU, nn.SELU, nn.Tanh, nn.Sigmoid]:
    #                 system = AutonomousSystem(nx, [hsize]*nlayers, act, linmap, sigmin, sigmax)
    #                 print(system(torch.zeroes(1, nx)))

    for nlayers in [1, 8]:
        for linmap, sigmin, sigmax in square_maps:
            for real in [True, False]:
                for act in [nn.ReLU, nn.SELU, nn.Tanh, nn.Sigmoid]:
                    system = AutonomousSystem(nx, [nx]*nlayers, act, linmap, sigmin, sigmax, real=real)
                    print(system(torch.zeros(1, nx)))


