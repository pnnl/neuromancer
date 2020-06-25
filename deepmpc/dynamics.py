"""
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
import torch.nn.functional as F
# local imports
import linear
import blocks
import constraints


def get_modules(model):
    return {name: module for name, module in model.named_modules()
            if len(list(module.named_children())) == 0}


# smart ways of initializing the weights?
class BlockSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fx, fy, fu=None, fd=None,
                 xou=torch.add, xod=torch.add, residual=False):
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

        if fu is not None:
            assert fu.in_features == nu, "Mismatch in control input function size"
            assert fu.out_features == nx, "Mismatch in control input function size"
        if fd is not None:
            assert fd.in_features == nd, "Mismatch in disturbance function size"
            assert fd.out_features == nx, "Mismatch in disturbance function size"

        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fx, self.fu, self.fd, self.fy = fx, fu, fd, fy
        # block operators
        self.xou = xou
        self.xod = xod
        # residual network
        self.residual = residual
        
        # Regularization Initialization
        self.xmin, self.xmax, self.umin, self.umax, self.uxmin, self.uxmax, self.dxmin, self.dxmax = self.con_init()
        self.Q_dx, self.Q_dx_ud, self.Q_con_x, self.Q_con_u, self.Q_sub = self.reg_weight_init()
      
        # slack variables for calculating constraints violations/ regularization error
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.s_sub = [0.0]*8

    def con_init(self):
        return [-0.2, 1.2, -0.2, 1.2, -1.2, 1.2, -1.2, 1.2] # default constraints for normalized dataset
        
    def reg_weight_init(self):
        return 0.0, 0.0, 0.0, 0.0, 0.2 # used for HW paper unconstrained
        # return [0.2] * 5  #  suggested new default for constrained HW

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)

#   TODO: shall we create separate module called constraints as a wrapper for the SSM?
#   by doing so we can avoid   con_init and reg_weight_init within SSM and move it out
#   moreover we could use the same constraints modules on policies and estimators

#    include regularization in each module in the framework
    def regularize(self, x_prev, x, u, fu, fd, N):

        # Barrier penalties
        self.sxmin = self.running_mean(self.sxmin, torch.mean(F.relu(-x + self.xmin)), N)
        self.sxmax = self.running_mean(self.sxmax, torch.mean(F.relu(x - self.xmax)), N)
        if u is not None:
            self.sumin = self.running_mean(self.sumin, torch.mean(F.relu(-u + self.umin)), N)
            self.sumax = self.running_mean(self.sumax, torch.mean(F.relu(u - self.umax)), N)
            self.dx_u = self.running_mean(self.dx_u,
                                          torch.mean(F.relu(-fu + self.uxmin) + F.relu(fu - self.uxmax)), N)
        # one step state residual penalty
        self.sdx_x = self.running_mean(self.sdx_x, torch.mean((x - x_prev)*(x - x_prev)), N)
        # penalties on max one-step infuence of controls and disturbances on states
        if fd is not None:
            self.dx_d = self.running_mean(self.dx_d,
                                          torch.mean(F.relu(-fd + self.dxmin) + F.relu(fd - self.dxmax)), N)
        # submodules regularization penalties
        self.s_sub = self.running_mean(self.s_sub, sum([k.reg_error() for k in
                                                        [self.fx, self.fu, self.fd, self.fy]
                                                        if hasattr(k, 'reg_error')]), N)

    def reg_error(self):
        error = sum([self.Q_con_x*self.sxmin, self.Q_con_x*self.sxmax,
                     self.Q_con_u*self.sumin, self.Q_con_u*self.sumax,
                     self.Q_dx*self.sdx_x, self.Q_dx_ud*self.dx_u,
                     self.Q_dx_ud*self.dx_d, self.Q_sub*self.s_sub])
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.s_sub = [0.0]*8
        return error

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
            u, fu, fd = None, None, None
            x_prev = x
            x = self.fx(x)
            if U is not None:
                u = U[i]
                fu = self.fu(U[i])
                x = self.xou(x, fu)
            if D is not None:
                fd = self.fd(D[i])
                x = x + self.xod(x, fd)
            if self.residual:
                x = x + x_prev
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, u, fu, fd, i+1)
        self.reset()
        return torch.stack(X), torch.stack(Y), self.reg_error()


class BlackSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fxud, fy):
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
        assert fxud.in_features == nx+nu+nd, "Mismatch in input function size"
        assert fxud.out_features == nx, "Mismatch in input function size"
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

    def con_init(self):
        return [-0.2, 1.2, -0.2, 1.2]  # default constraints for normalized dataset

    def reg_weight_init(self):
        return [0] * 3 + [0.2]  # uncostrained, only regularization
        # return [0.2] * 4 # suggested new default for constrained HW

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)

    # TODO: update regularization once we agree upon its form
    def regularize(self, x_prev, x, u, N):
        # Barrier penalties
        self.sxmin = self.running_mean(self.sxmin, torch.mean(F.relu(-x + self.xmin)), N)
        self.sxmax = self.running_mean(self.sxmax, torch.mean(F.relu(x - self.xmax)), N)
        if u is not None:
            self.sumin = self.running_mean(self.sumin, torch.mean(F.relu(-u + self.umin)), N)
            self.sumax = self.running_mean(self.sumax, torch.mean(F.relu(u - self.umax)), N)
        # one step state residual penalty
        self.sdx_x = self.running_mean(self.sdx_x, torch.mean((x - x_prev) * (x - x_prev)), N)
        # submodules regularization penalties
        self.s_sub = self.running_mean(self.s_sub, sum([k.reg_error() for k in
                                                        [self.fxud, self.fy]
                                                        if hasattr(k, 'reg_error')]), N)

    def reg_error(self):
        error = sum([self.Q_con_x * self.sxmin, self.Q_con_x * self.sxmax,
                     self.Q_con_u * self.sumin, self.Q_con_u * self.sumax,
                     self.Q_dx * self.sdx_x, self.Q_sub * self.s_sub])
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.s_sub = [0.0] * 6
        return error

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
            u = None
            xi = x
            if U is not None:
                u = U[i]
                xi = torch.cat([xi, u], dim=1)
            if D is not None:
                d = D[i]
                xi = torch.cat([xi, d], dim=1)
            x_prev = x
            x = self.fxud(xi)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, u, i+1)
        self.reset()
        return torch.stack(X), torch.stack(Y), self.reg_error()



# TODO: differentiate autonomous and non-autonomous odes
# and odes with and without disturbances
"""
gray-box ODE for training with customly defined nonlinmap
TODO: create new file NeuralODEs.py with differentiable ODE priors
then use them as specifi nonlinear and linear maps
e.g., nonlinmap = [CSTR, LorentzSystem]
linmap = identity not being learned
"""
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

    def con_init(self):
        return [-0.2, 1.2, -0.2, 1.2]  # default constraints for normalized dataset

    def reg_weight_init(self):
        return [0] * 3 + [0.2]  # uncostrained, only regularization
        # return [0.2] * 4 # suggested new default for constrained HW

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)

    # TODO: update regularization once we agree upon its form
    def regularize(self, x_prev, x, u, N):
        # Barrier penalties
        self.sxmin = self.running_mean(self.sxmin, torch.mean(F.relu(-x + self.xmin)), N)
        self.sxmax = self.running_mean(self.sxmax, torch.mean(F.relu(x - self.xmax)), N)
        self.sumin = self.running_mean(self.sumin, torch.mean(F.relu(-u + self.umin)), N)
        self.sumax = self.running_mean(self.sumax, torch.mean(F.relu(u - self.umax)), N)
        # one step state residual penalty
        self.sdx_x = self.running_mean(self.sdx_x, torch.mean((x - x_prev) * (x - x_prev)), N)
        # submodules regularization penalties
        self.s_sub = self.running_mean(self.s_sub, sum([k.reg_error() for k in
                                                        [self.fxud, self.fy]
                                                        if hasattr(k, 'reg_error')]), N)

    def reg_error(self):
        error = sum([self.Q_con_x * self.sxmin, self.Q_con_x * self.sxmax,
                     self.Q_con_u * self.sumin, self.Q_con_u * self.sumax,
                     self.Q_dx * self.sdx_x, self.Q_sub * self.s_sub])
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.s_sub = [0.0] * 6
        return error

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





def blackbox(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
    """
    black box state space model for training
    """
    fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx]*n_layers,
                     bias=args.bias, Linear=linmap, skip=1)
    fy = linmap(nx, ny, bias=args.bias)
    return BlackSSM(nx, nu, nd, ny, fxud, fy)

def blocknlin(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
    """
    block nonlinear state space model for training
    """
    fx = nonlinmap(nx, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linmap, skip=1)
    fy = linmap(nx, ny, bias=args.bias)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)

def hammerstein(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
    """
    hammerstein state space model for training
    """
    fx = linmap(nx, nx, bias=args.bias)
    fy = linmap(nx, ny, bias=args.bias)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)

def hammerstein_bilinearfu(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
    """
    hammerstein state space model for training with bilinear input
    suitable for building thermal dynamics
    """
    fx = linmap(nx, nx, bias=args.bias)
    fy = linmap(nx, ny, bias=args.bias)
    # TODO: customize for building models separate mass flows and temperatures
    # TODO: this would probably require to modify BlockSSM or handle u in fu
    fu = blocks.BilinearTorch(nu, nx, bias=args.bias, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)

def hw(args, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2):
    """
    hammerstein-weiner state space model for training
    """
    fx = linmap(nx, nx, bias=args.bias)
    fy = nonlinmap(nx, ny, bias=args.bias, hsizes=[nx]*n_layers, Linear=linmap, skip=1)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=args.bias, hsizes=[nx]*n_layers, Linear=linear.Linear, skip=1) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)


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

    # block SSM
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = blocks.MLP(nx, ny, hsizes=[64, 64, 64])
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    output = model(x, U, D)
    # black box SSM
    fxud = blocks.MLP(nx+nu+nd, nx, hsizes=[64, 64, 64])
    fy = linear.Linear(nx, ny)
    model = BlackSSM(nx, nu, nd, ny, fxud, fy)
    output = model(x, U, D)
    fxud = blocks.RNN(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model = BlackSSM(nx, nu, nd, ny, fxud, fy)
    output = model(x, U, D)

