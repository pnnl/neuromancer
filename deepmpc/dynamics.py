"""
Classes in the module implement the Component interface.
# TODO: Describe Component interface
# TODO: Re-implement RNN state preservation for open loop simulation


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

# ecosystem imports
import slim

# local imports
import deepmpc.blocks as blocks


class BlockSSM(nn.Module):
    def __init__(self, fx, fy, fu=None, fd=None,
                 xou=torch.add, xod=torch.add, residual=False, name='block_ssm',
                 input_keys=dict()):
        """

        generic structured system dynamics:
        # x+ = fx(x) o fu(u) o fd(d)
        # y =  fy(x)
        """
        super().__init__()
        self.fx, self.fy, self.fu, self.fd = fx, fy, fu, fd
        self.check_features()
        self.name, self.residual = name, residual
        self.input_keys = self.keys(input_keys)
        # block operators
        self.xou = xou
        self.xod = xod

    @staticmethod
    def keys(input_keys):
        default_keys = {'x0': 'x0', 'Yf': 'Yf', 'Uf': 'Uf', 'Df': 'Df'}
        new_keys = {**default_keys, **input_keys}
        return [new_keys['x0'], new_keys['Yf'], new_keys['Uf'], new_keys['Df']]

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        x_in, y_out, u_in, d_in = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y, FD, FU = [], [], [], []

        x = data[x_in]
        for i in range(nsteps):
            x_prev = x
            x = self.fx(x)
            if self.fu is not None:
                fu = self.fu(data[u_in][i])
                x = self.xou(x, fu)
                FU.append(fu)
            if self.fd is not None:
                fd = self.fd(data[d_in][i])
                x = x + self.xod(x, fd)
                FD.append(fd)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            X.append(x)
            Y.append(y)

        return {f'X_pred_{self.name}': torch.stack(X), f'Y_pred_{self.name}': torch.stack(Y),
                f'reg_error_{self.name}': self.reg_error(),
                f'fU_{self.name}': torch.stack(FU) if FU else None,
                f'fD_{self.name}': torch.stack(FD) if FD else None}

    def check_features(self):
        self.nx, self.ny = self.fx.in_features, self.fy.out_features
        self.nu = self.fu.in_features if self.fu is not None else 0
        self.nd = self.fd.in_features if self.fd is not None else 0
        assert self.fx.in_features == self.fx.out_features, 'State transition must have same input and output dimensions'
        assert self.fy.in_features == self.fx.out_features, 'Output map must have same input size as output size of state transition'
        if self.fu is not None:
            assert self.fu.out_features == self.fx.out_features, 'Dimension mismatch between input and state transition'
        if self.fd is not None:
            assert self.fd.out_features == self.fx.out_features, 'Dimension mismatch between disturbance and state transition'

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class BlackSSM(nn.Module):
    def __init__(self, fxud, fy, name='black_ssm', input_keys=dict(), residual=False):
        """
        atomic black box state space model generic unstructured system dynamics:
        # x+ = fxud(x,u,d)
        # y =  fy(x)
        """
        super().__init__()
        self.fxud, self.fy = fxud, fy
        self.name, self.residual = name, residual
        self.input_keys = BlockSSM.keys(input_keys)

    def forward(self, data):
        """
        """
        x_in, y_out, u_in, d_in = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y = [], []

        x = data[x_in]
        for i in range(nsteps):
            x_prev = x
            # Concatenate x with u and d if they are available in the dataset.
            x = torch.cat([x] + [data[k][i] for k in [u_in, d_in] if k in data], dim=1)
            x = self.fxud(x)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        return {f'X_pred_{self.name}': torch.stack(X),
                f'Y_pred_{self.name}': torch.stack(Y),
                f'reg_error_{self.name}': self.reg_error()}

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])

    def check_features(self):
        self.nx, self.ny = self.fxud.out_features, self.fy.out_features
        assert self.fxud.out_features == self.fy.in_features, 'Output map must have same input size as output size of state transition'


def blackbox(bias, linmap, nonlinmap, datadims, n_layers=2,
             activation=nn.GELU, name='blackbox', input_keys=dict(), residual=False):
    """
    black box state space model for training
    """
    xkey, ykey, ukey, dkey = BlockSSM.keys(input_keys)
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0
    fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx]*n_layers,
                     bias=bias, Linear=linmap, nonlin=activation)
    fy = linmap(nx, ny, bias=bias)
    return BlackSSM(fxud, fy, name=name, input_keys=input_keys, residual=residual)


def blocknlin(bias, linmap, nonlinmap, datadims, n_layers=2,
              activation=nn.GELU, name='blocknonlin', input_keys=dict(), residual=False):
    """
    block nonlinear state space model for training
    """

    xkey, ykey, ukey, dkey = BlockSSM.keys(input_keys)
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0
    fx = nonlinmap(nx, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linmap, nonlin=activation)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nd != 0 else None
    return BlockSSM(fx, fy, fu=fu, fd=fd, name=name, input_keys=input_keys, residual=residual)


def hammerstein(bias, linmap, nonlinmap, datadims, n_layers=2,
                activation=nn.GELU, name='hammerstein', input_keys=dict(), residual=False):
    """
    hammerstein state space model for training
    """
    xkey, ykey, ukey, dkey = BlockSSM.keys(input_keys)
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0
    fx = linmap(nx, nx, bias=bias)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nd != 0 else None
    return BlockSSM(fx, fy, fu=fu, fd=fd, name=name, input_keys=input_keys, residual=residual)


def hw(bias, linmap, nonlinmap, datadims, n_layers=2,
                activation=nn.GELU, name='hw', input_keys=dict(), residual=False):
    """
    hammerstein-weiner state space model for training
    """
    xkey, ykey, ukey, dkey = BlockSSM.keys(input_keys)
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0
    fx = linmap(nx, nx, bias=bias)
    fy = nonlinmap(nx, ny, bias=bias, hsizes=[nx]*n_layers, Linear=linmap, nonlin=activation)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=slim.Linear, nonlin=activation) if nd != 0 else None
    return BlockSSM(fx, fy, fu=fu, fd=fd, name=name, input_keys=input_keys, residual=residual)


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
    datadims = {'x0': (nx,), 'Uf': (N, nu), 'Df': (N, nd), 'Yf': (N, ny)}
    # block SSM
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = blocks.MLP(nx, ny, hsizes=[64, 64, 64])
    model = BlockSSM(fx, fy, fu, fd)
    model = BlockSSM(fx, fy, fu, fd)
    output = model(data)
    # black box SSM
    fxud = blocks.MLP(nx+nu+nd, nx, hsizes=[64, 64, 64])
    fy = slim.Linear(nx, ny)
    model = BlackSSM(fxud, fy)
    output = model(data)
    fxud = blocks.RNN(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model = BlackSSM(fxud, fy)
    output = model(data)

    data = {'x0_new': x, 'Uf': U, 'Df': D, 'Yf_fresh': D}
    datadims = {'x0_new': (nx,), 'Uf': (N, nu), 'Df': (N, nd), 'Yf_fresh': (N, ny)}
    # block SSM
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = blocks.MLP(nx, ny, hsizes=[64, 64, 64])
    model = BlockSSM(fx, fy, fu, fd, input_keys={'x0': 'x0_new', 'Yf': 'Yf_fresh'})
    model = BlockSSM(fx, fy, fu, fd, input_keys={'x0': 'x0_new', 'Yf': 'Yf_fresh'})
    output = model(data)
    # black box SSM
    fxud = blocks.MLP(nx + nu + nd, nx, hsizes=[64, 64, 64])
    fy = slim.Linear(nx, ny)
    model = BlackSSM(fxud, fy, input_keys={'x0': 'x0_new', 'Yf': 'Yf_fresh'})
    output = model(data)
    fxud = blocks.RNN(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model = BlackSSM(fxud, fy, input_keys={'x0': 'x0_new', 'Yf': 'Yf_fresh'})
    output = model(data)