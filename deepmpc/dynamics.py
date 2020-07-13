"""
Classes in the module implement the Component interface.
TODO: Describe Component interface

state space models (SSM)
x: states
y: predicted outputs
u: control inputs
d: uncontrolled inputs (measured disturbances)

unstructured dynamical models:
x+ = f(x,u,d)
y =  fy(x)

Block dynamical models:
x+ = fx(x) o fu(u) o fd(d)
y =  fy(x)

o = operator, e.g., +, or *
any operation perserving dimensions
"""
# pytorch imports
import torch
import torch.nn as nn
# local imports
import linear
import blocks


def get_modules(model):
    return {name: module for name, module in model.named_modules()
            if len(list(module.named_children())) == 0}


def check_keys(k1, k2):
    assert k1 - k2 == set(), \
        f'Missing values in dataset. Input_keys: {k1}, data_keys: {k2}'


class BlockSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fx, fy, fu=None, fd=None,
                 xou=torch.add, xod=torch.add, residual=False, name='block_ssm'):
        """
        atomic block state space model
        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fx: function R^{nx} -> R^(nx}
        :param fu: function R^{nu} -> R^(nx}
        :param fd: function R^{nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}
        :param xou: Shape preserving binary operator (e.g. +, -, *)
        :param xod: Shape preserving binary operator (e.g. +, -, *)

        generic structured system dynamics:   
        # x+ = fx(x) o fu(u) o fd(d)
        # y =  fy(x)
        """
        super().__init__()
        assert fx.in_features == nx, "Mismatch in input function size"
        assert fx.out_features == nx, "Mismatch in input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"
        self.input_keys = {'Yf', 'x0'}
        if fu is not None:
            assert fu.in_features == nu, "Mismatch in control input function size"
            assert fu.out_features == nx, "Mismatch in control input function size"
        if fd is not None:
            assert fd.in_features == nd, "Mismatch in disturbance function size"
            assert fd.out_features == nx, "Mismatch in disturbance function size"

        self.name = name
        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fx, self.fu, self.fd, self.fy = fx, fu, fd, fy
        # block operators
        self.xou = xou
        self.xod = xod
        # residual network
        self.residual = residual

    def reg_error(self):
        # submodules regularization penalties
        return sum([k.reg_error() for k in [self.fx, self.fu, self.fd, self.fy] if hasattr(k, 'reg_error')])

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'reset') and mod is not self:
                mod.reset()

    def forward(self, data):
        """
        """
        check_keys(self.input_keys, set(data.keys()))
        nsteps = data['Yf'].shape[0]
        X, Y = [], []
        x = data['x0']
        for i in range(nsteps):
            x_prev = x
            x = self.fx(x)
            if 'Uf' in data:
                fu = self.fu(data['Uf'][i])
                x = self.xou(x, fu)
            if 'Df' in data:
                fd = self.fd(data['Df'][i])
                x = x + self.xod(x, fd)
            if self.residual:
                x = x + x_prev
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        self.reset()
        return {'X_pred': torch.stack(X), 'Y_pred': torch.stack(Y), f'{self.name}_reg_error': self.reg_error()}


class BlackSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fxud, fy, name='black_ssm'):
        """
        atomic black box state space model
        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fxud: function R^{nx+nu+nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}

        generic unstructured system dynamics:
        # x+ = fxud(x,u,d)
        # y =  fy(x)
        """
        super().__init__()
        self.name = name
        assert fxud.in_features == nx+nu+nd, "Mismatch in input function size"
        assert fxud.out_features == nx, "Mismatch in input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"
        self.input_keys = {'Yf', 'x0'}
        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fxud, self.fy = fxud, fy

    def reg_error(self):
        # submodules regularization penalties
        return sum([k.reg_error() for k in [self.fxud, self.fy] if hasattr(k, 'reg_error')])

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'reset') and mod is not self:
                mod.reset()

    def forward(self, data):
        """
        """
        check_keys(self.input_keys, set(data.keys()))
        nsamples = data['Yf'].shape[0]
        X, Y = [], []
        x = data['x0']
        for i in range(nsamples):
            xi = x
            if 'Uf' in data:
                u = data['Uf'][i]
                xi = torch.cat([xi, u], dim=1)
            if 'Df' in data:
                d = data['Df'][i]
                xi = torch.cat([xi, d], dim=1)
            x = self.fxud(xi)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        self.reset()
        return {'X_pred': torch.stack(X), 'Y_pred': torch.stack(Y), f'{self.name}_reg_error': self.reg_error()}


def blackbox(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2, name='blackbox'):
    """
    black box state space model for training
    """
    fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx]*n_layers,
                     bias=bias, Linear=linmap)
    fy = linmap(nx, ny, bias=bias)
    return BlackSSM(nx, nu, nd, ny, fxud, fy, name=name)


def blocknlin(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2, name='blocknlin'):
    """
    block nonlinear state space model for training
    """
    fx = nonlinmap(nx, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linmap)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, name=name)


def hammerstein(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2, name='hammerstein'):
    """
    hammerstein state space model for training
    """
    fx = linmap(nx, nx, bias=bias)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, name=name)


def hw(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2, name='hw'):
    """
    hammerstein-weiner state space model for training
    """
    fx = linmap(nx, nx, bias=bias)
    fy = nonlinmap(nx, ny, bias=bias, hsizes=[nx]*n_layers, Linear=linmap)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, name=name)


ssm_models_atoms = [BlockSSM, BlackSSM]
ssm_models_train = [blackbox, hammerstein, hw, blocknlin]

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    U = torch.rand(N, samples, nu)
    D = torch.rand(N, samples, nd)

    data = {'x0': x, 'Uf': U, 'Df': D, 'Yf': D}
    # block SSM
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = blocks.MLP(nx, ny, hsizes=[64, 64, 64])
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    output = model(data)
    # black box SSM
    fxud = blocks.MLP(nx+nu+nd, nx, hsizes=[64, 64, 64])
    fy = linear.Linear(nx, ny)
    model = BlackSSM(nx, nu, nd, ny, fxud, fy)
    output = model(data)
    fxud = blocks.RNN(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model = BlackSSM(nx, nu, nd, ny, fxud, fy)
    output = model(data)

#
# def hammerstein_bilinearfu(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
#     """
#     hammerstein state space model for training with bilinear input
#     suitable for building thermal dynamics
#     """
#     fx = linmap(nx, nx, bias=args.bias)
#     fy = linmap(nx, ny, bias=args.bias)
#     # TODO: customize for building models separate mass flows and temperatures
#     # TODO: this would probably require to modify BlockSSM or handle u in fu
#     fu = blocks.BilinearTorch(nu, nx, bias=args.bias, Linear=linear.Linear) if nu != 0 else None
#     fd = nonlinmap(nd, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
#     return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)

# # TODO: implement multistep neural network
# # TODO: exactly copy their implementation for comparison
# # https://arxiv.org/abs/1801.01236
# # https://github.com/maziarraissi/MultistepNNs/blob/master/Multistep_NN.py
#
# # TODO: finish and test our version of multistep method
# # instead of fixed values of parameters we will learn them
# class MultistepBlack(BlackSSM):
#     def __init__(self, nx, nu, nd, ny, fxud, fy, M=1):
#         """
#         Implements generic explicit multistep formula of Adams–Bashforth methods
#         x_k = x_k-1 + h*(b_1*f(x_k-1) +  b_2*f(x_k-2) + ... b_M*f(x_k-M))
#         sum(b_i) = 1, for all i = 1,...,M
#         h = step size
#
#         # TODO: implement stability condition on coefficients
#         A linear multistep method is zero-stable if and only if the root condition is satisfied
#         # https://en.wikipedia.org/wiki/Linear_multistep_method
#         """
#         super().__init__(nx, nu, nd, ny, fxud, fy)
#         self.M = M  # number of steps of the multistep method
#         self.h = 1  # step size, default = 1
#         self.Bsum_error = 0.0 # regularization error of the multistep coefficient constraints
#
#     # TODO: move this formulat to blocks?
#     def AdamsBashforth(self, X, U, D, M):
#         # computing Adams Bashforth residual
#         self.Beta = nn.Parameter(torch.randn(1, M))
#         for j in range(M):
#             u = None
#             x = X[j]
#             xi = x
#             if U is not None:
#                 u = U[j]
#                 xi = torch.cat([xi, u], dim=1)
#             if D is not None:
#                 d = D[j]
#                 xi = torch.cat([xi, d], dim=1)
#             dx = self.h*(self.Beta[j]*self.fxud(xi)) if j == 0 \
#                 else dx + self.h*(self.Beta[j]*self.fxud(xi))
#         x = x + dx
#         self.Bsum_error = (1-sum(self.Beta))*(1-sum(self.Beta))
#         return x, u
#
#     def forward(self, x, U=None, D=None, nsamples=1):
#         """
#         """
#         if U is not None:
#             nsamples = U.shape[0]
#         elif D is not None:
#             nsamples = D.shape[0]
#
#         X, Y = [], []
#
#         for i in range(nsamples):
#             x_prev = x
#             if i == 0:
#                 Mi = 0
#                 Xi = x
#                 Ui = U[i] if U is not None else None
#                 Di = D[i] if D is not None else None
#             elif i<self.M:
#                 Mi = i
#                 Xi = Xi.append(x)
#                 Ui = U[0:i] if U is not None else None
#                 Di = D[0:i] if D is not None else None
#             else:
#                 Mi = self.M
#                 Xi = Xi.append(x)
#                 Xi = Xi[i-self.M:i]
#                 Ui = U[i-self.M:i] if U is not None else None
#                 Di = D[i-self.M:i] if D is not None else None
#             x, u = self.AdamsBashforth(Xi, Ui, Di, Mi)
#             y = self.fy(x)
#             X.append(x)
#             Y.append(y)
#             self.regularize(x_prev, x, u, i + 1)
#
#         self.reset()
#         return torch.stack(X), torch.stack(Y), self.reg_error()+self.Bsum_error
#
#
#
# # TODO: differentiate autonomous and non-autonomous odes
# # and odes with and without disturbances
# """
# gray-box ODE for training with customly defined nonlinmap
# TODO: create new file NeuralODEs.py with differentiable ODE priors
# then use them as specifi nonlinear and linear maps
# e.g., nonlinmap = [CSTR, LorentzSystem]
# linmap = identity not being learned
# """
class GrayODE(nn.Module):
    def __init__(self, nx, nu, nd, ny, fxud, fy):
        """
        atomic gray-box ODE model
        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fxud: function R^{nx+nu+nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}

        generic discrete-time ODE system dynamics:
        # x+ = fxud(x,u,d)
        # y =  fy(x)
        """
        super().__init__()
        assert fxud.in_features == nx+nu+nd, "Mismatch in input function size"
        assert fxud.nx == nx, "Mismatch in input function size"
        assert fxud.nu == nu, "Mismatch in input function size"
        assert fxud.nd == nd, "Mismatch in input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"

        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fxud, self.fy = fxud, fy
        # TODO: if fy is None, then use identity without gradient

        # Regularization Initialization
        self.xmin, self.xmax, self.umin, self.umax = self.con_init()
        self.Q_dx, self.Q_con_x, self.Q_con_u, self.Q_sub = self.reg_weight_init()

        # slack variables for calculating constraints violations/ regularization error
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.s_sub = [0.0] * 8

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'reset') and mod is not self:
                mod.reset()

    def forward(self, x, U=None, D=None, nsamples=1):
        """
        """
        if U is not None:
            nsamples = U.shape[0]
        elif D is not None:
            nsamples = D.shape[0]

        X, Y = [], []
        for i in range(nsamples):
            u = U[i] if U is not None else None
            d = D[i] if D is not None else None
            x_prev = x
            x = self.fxud(x, u, d)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, u, i+1)
        self.reset()
        return torch.stack(X), torch.stack(Y), self.reg_error()
