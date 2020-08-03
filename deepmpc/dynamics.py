"""
Classes in the module implement the Component interface.
# TODO: Describe Component interface
# TODO: update dynamics model to be compiant with dataset.dims input just like estim and policy
# TODO: we may want to specify nsims explicitly and not via Yf - e.g. for control dataset Yf may not be available
# TODO: nsims in the dataset
# TODO: allow arbitraty input_keys for the user
# TODO: block SSM work if we have D but not U - see line 119
# TODO: customize for building models separate mass flows and temperatures
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
# local imports
import linear
import blocks


def get_modules(model):
    return {name: module for name, module in model.named_modules()
            if len(list(module.named_children())) == 0}


def check_keys(k1, k2):
    assert set(k1) - set(k2) == set(), \
        f'Missing values in dataset. Input_keys: {set(k1)}, data_keys: {set(k2)}'


class BlockSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fx, fy, fu=None, fd=None,
                 xou=torch.add, xod=torch.add, residual=False,
                 input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred', 'fU_pred', 'fD_pred'],
                 name='block_ssm'):
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
        :param input_keys: (list of str) input keys in expected order: nsim, x0, U, D,
        :param output_keys: (list of str) output keys in expected order: X_pred, Y_pred, fU_pred, fD_pred

        generic structured system dynamics:   
        # x+ = fx(x) o fu(u) o fd(d)
        # y =  fy(x)
        """
        super().__init__()
        self.name = name
        assert fx.in_features == nx, "Mismatch in input function size"
        assert fx.out_features == nx, "Mismatch in input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"
        assert 2 <= len(input_keys) <= 4, "Mismatch in number of input keys, requires at least 2 keys and at most 4 keys"
        assert len(output_keys) == 4, f"Mismatch in number of output keys, requires 4, provided {len(output_keys)}"
        assert 'Yf' in input_keys, "Need 'Yf' in input_keys to determine nsim"
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.output_keys.append(f'{self.name}_reg_error')
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

    def reg_error(self):
        # submodules regularization penalties
        return sum([k.reg_error() for k in [self.fx, self.fu, self.fd, self.fy] if hasattr(k, 'reg_error')])

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        check_keys(self.input_keys, set(data.keys()))
        nsteps = data[self.input_keys[0]].shape[0]
        X, Y, FD, FU = [], [], [], []
        x = data[self.input_keys[1]]
        for i in range(nsteps):
            x_prev = x
            x = self.fx(x)
            if len(self.input_keys) > 2 and self.input_keys[2] in data:
                fu = self.fu(data[self.input_keys[2]][i])
                x = self.xou(x, fu)
                FU.append(fu)
            # DANGER!!: This hackey solution will break if you have D and not U
            if len(self.input_keys) > 3 and self.input_keys[3] in data:
                fd = self.fd(data[self.input_keys[3]][i])
                x = x + self.xod(x, fd)
                FD.append(fd)
            if self.residual:
                x = x + x_prev
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        return {self.output_keys[0]: torch.stack(X), self.output_keys[1]: torch.stack(Y),
                       self.output_keys[4]: self.reg_error(),
                       self.output_keys[2]: torch.stack(FU) if not not FU else None,
                       self.output_keys[3]: torch.stack(FD) if not not FD else None}


class BlackSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fxud, fy, input_keys=['Yf', 'x0', 'Uf', 'Df'],
                 output_keys=['X_pred', 'Y_pred'], name='black_ssm'):
        """
        atomic black box state space model
        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fxud: function R^{nx+nu+nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}
        :param input_keys: (list of str) input keys in expected order: Y, x0, U, D
        :param output_keys: (list of str) output keys in expected order: X_pred, Y_pred

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
        assert 2 <= len(input_keys) <= 4, "Mismatch in number of input keys, requires at least 2 keys and at most 4 keys"
        assert len(output_keys) == 2, "Mismatch in number of output keys, requires 2"
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.output_keys.append(f'{self.name}_reg_error')
        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fxud, self.fy = fxud, fy

    def reg_error(self):
        # submodules regularization penalties
        return sum([k.reg_error() for k in [self.fxud, self.fy] if hasattr(k, 'reg_error')])

    def forward(self, data):
        """
        """
        check_keys(self.input_keys, set(data.keys()))
        nsteps = data[self.input_keys[0]].shape[0]
        X, Y = [], []
        x = data[self.input_keys[1]]
        for i in range(nsteps):
            xi = x
            if len(self.input_keys) > 2 and self.input_keys[2] in data:
                u = data[self.input_keys[2]][i]
                xi = torch.cat([xi, u], dim=1)
            if len(self.input_keys) > 3 and self.input_keys[3] in data:
                d = data[self.input_keys[3]][i]
                xi = torch.cat([xi, d], dim=1)
            x = self.fxud(xi)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
        return {self.output_keys[0]: torch.stack(X), self.output_keys[1]: torch.stack(Y), self.output_keys[2]: self.reg_error()}


def blackbox(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2,
             input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred'], name='blackbox'):
    """
    black box state space model for training
    """

    fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx]*n_layers,
                     bias=bias, Linear=linmap)
    fy = linmap(nx, ny, bias=bias)
    return BlackSSM(nx, nu, nd, ny, fxud, fy, input_keys=input_keys, output_keys=output_keys, name=name)


def blocknlin(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2,
              input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred', 'fU_pred', 'fD_pred'], name='blocknlin'):
    """
    block nonlinear state space model for training
    """
    fx = nonlinmap(nx, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linmap)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, input_keys=input_keys, output_keys=output_keys, name=name)


def hammerstein(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2,
                input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred', 'fU_pred', 'fD_pred'], name='hammerstein'):
    """
    hammerstein state space model for training
    """
    fx = linmap(nx, nx, bias=bias)
    fy = linmap(nx, ny, bias=bias)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, input_keys=input_keys, output_keys=output_keys, name=name)


def hw(bias, linmap, nonlinmap, nx, nu, nd, ny, n_layers=2,
       input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred', 'fU_pred', 'fD_pred'], name='hw'):
    """
    hammerstein-weiner state space model for training
    """
    fx = linmap(nx, nx, bias=bias)
    fy = nonlinmap(nx, ny, bias=bias, hsizes=[nx]*n_layers, Linear=linmap)
    fu = nonlinmap(nu, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nu != 0 else None
    fd = nonlinmap(nd, nx, bias=bias, hsizes=[nx]*n_layers, Linear=linear.Linear) if nd != 0 else None
    return BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd, input_keys=input_keys, output_keys=output_keys, name=name)


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
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd,
                     input_keys=['Yf', 'x0', 'Uf', 'Df'],
                     output_keys=['X_pred', 'Y_pred', 'fU_pred', 'fD_pred'])
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    output = model(data)
    # black box SSM
    fxud = blocks.MLP(nx+nu+nd, nx, hsizes=[64, 64, 64])
    fy = linear.Linear(nx, ny)
    model = BlackSSM(nx, nu, nd, ny, fxud, fy,
                     input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred'])
    output = model(data)
    fxud = blocks.RNN(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model = BlackSSM(nx, nu, nd, ny, fxud, fy,
                     input_keys=['Yf', 'x0', 'Uf', 'Df'], output_keys=['X_pred', 'Y_pred'])
    output = model(data)


