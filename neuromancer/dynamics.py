"""
State space models (SSMs) for dynamical modeling.

Nomenclature:

    + x: states
    + y: predicted outputs
    + u: control inputs
    + d: uncontrolled inputs (measured disturbances)

Unstructured (blackbox) dynamical models:
    + :math:`x_{t+1} = f(x_t,u_t,d_t) \odot f_e(x_t)`
    + :math:`y_{t} =  f_y(x_t)`
    + :math:`odot` is some operator acting on elements, e.g. + or *

Block-structured dynamical models:
    + :math:`x_{t+1} = f_x(x_t) \odot f_u(u_t) \odot f_d(d_t) \odot f_e(x_t)`
    + :math:`y_t =  f_y(x_t)`

Block components:
    + :math:`f_e` is an error model
    + :math:`f_{xudy}` is a nominal model
    + :math:`f_x` is the main state transition dynamics model
    + :math:`f_y` is the measurement function mapping latent dynamics to observations
    + :math:`f_u` are input dynamics
    + :math:`f_d` are disturbance dynamics
    + :math:`f_{yu}` models more direct effects of input on observations
"""

import torch
import torch.nn as nn

import slim

import neuromancer.blocks as blocks
from neuromancer.component import Component


class BlockSSM(Component):
    DEFAULT_INPUT_KEYS = ['x0', 'Yf', 'Uf', 'Df']
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "fU", "fD", "fE", "reg_error"]

    def __init__(self, fx, fy, fu=None, fd=None, fe=None, fyu=None,
                 xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add, residual=False, name='block_ssm',
                 input_keys=dict()):
        """
        Block structured system dynamics:

        :param fx: (nn.Module) State transition function
        :param fy: (nn.Module) Observation function
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param fe: (nn.Module) Error term via state augmentation
        :param fyd: (nn.Module) Input-Observation function
        :param xou: (callable) Elementwise tensor op
        :param xod: (callable) Elementwise tensor op
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        """
        assert isinstance(input_keys, dict), \
            f"BlockSSM input_keys must be dict, type is {type(input_keys)}"
        super().__init__(
            input_keys,
            BlockSSM.DEFAULT_OUTPUT_KEYS,
            name,
        )

        self.fx, self.fy, self.fu, self.fd, self.fe, self.fyu = fx, fy, fu, fd, fe, fyu
        self.nx, self.ny, self.nu, self.nd = (
            self.fx.in_features,
            self.fy.out_features,
            self.fu.in_features if fu is not None else None,
            self.fd.in_features if fd is not None else None
        )

        in_features = self.nx
        in_features = in_features + self.fu.in_features if fu is not None else in_features
        in_features = in_features + self.fd.in_features if fd is not None else in_features
        self.in_features = in_features
        self.out_features = self.fy.out_features

        self.check_features()
        self.residual = residual

        # block operators
        self.xou = xou
        self.xod = xod
        self.xoe = xoe
        self.xoyu = xoyu

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

    @staticmethod
    def keys(input_keys):
        """
        Overwrite canonical expected input keys with alternate names

        :param input_keys: (dict {str:str}) Mapping canonical expected input keys to alternate names
        :return: (list [str]) List of input keys
        """
        default_keys = {'x0': 'x0', 'Yf': 'Yf', 'Uf': 'Uf', 'Df': 'Df'}
        new_keys = {**default_keys, **input_keys}
        return [new_keys['x0'], new_keys['Yf'], new_keys['Uf'], new_keys['Df']]

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        print(data.keys())
        x_in, y_out, u_in, d_in = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y, FD, FU, FE = [], [], [], [], []
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
                x = self.xod(x, fd)
                FD.append(fd)
            if self.fe is not None:
                fe = self.fe(x)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(data[u_in][i])
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)
        output = dict()
        for tensor_list, name in zip([X, Y, FU, FD, FE],
                                     ['X_pred', 'Y_pred', 'fU', 'fD', 'fE']):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        output['reg_error'] = self.reg_error()
        return output

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class BlackSSM(Component):
    DEFAULT_INPUT_KEYS = ['x0', 'Yf', 'Uf', 'Df']
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "fE", "reg_error"]

    def __init__(self, fxud, fy, fe=None, fyu=None, xoe=torch.add, xoyu=torch.add, name='black_ssm', input_keys=dict(), residual=False):
        """
        Black box state space model with unstructured system dynamics:

        :param fxud: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param fyd: (nn.Module) Input-Observation function
        :param fe: (nn.Module) Error term via state augmentation
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param residual: (bool) Whether to make recurrence in state space model residual

        """
        assert isinstance(input_keys, dict), \
            f"BlackSSM input_keys must be dict, type is {type(input_keys)}"
        super().__init__(
            input_keys,
            BlackSSM.DEFAULT_OUTPUT_KEYS,
            name,
        )
        self.fxud, self.fy, self.fe, self.fyu = fxud, fy, fe, fyu
        self.nx, self.ny = self.fxud.out_features, self.fy.out_features
        self.residual = residual
        self.xoe = xoe
        self.xoyu = xoyu
        self.in_features = self.fxud.out_features
        self.out_features = self.fy.out_features

    def forward(self, data):
        """
        """
        x_in, y_out, u_in, d_in = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y, FE = [], [], []

        x = data[x_in]
        for i in range(nsteps):
            x_prev = x
            # Concatenate x with u and d if they are available in the dataset.
            x = torch.cat([x] + [data[k][i] for k in [u_in, d_in] if k in data], dim=1)
            x = self.fxud(x)
            if self.fe is not None:
                fe = self.fe(x)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(data[u_in][i])
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)
        output = dict()
        for tensor_list, name in zip([X, Y, FE],
                                     ['X_pred', 'Y_pred', 'fE']):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        output['reg_error'] = self.reg_error()
        return output

    def reg_error(self):
        """

        :return: 0-dimensional torch.Tensor
        """
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])

    def check_features(self):
        self.nx, self.ny = self.fxud.out_features, self.fy.out_features
        assert self.fxud.out_features == self.fy.in_features, 'Output map must have same input size as output size of state transition'


class TimeDelayBlockSSM(BlockSSM):
    def __init__(self, fx, fy, fu=None, fd=None, fe=None,
                 xou=torch.add, xod=torch.add, xoe=torch.add, residual=False, name='block_ssm',
                 input_keys=dict(), timedelay=0):
        """
        generic structured time delayed system dynamics:
        T < nsteps

        Option 1 - fixed time delays - IMPLEMENTED
        x_k+1 = fx(x_k, ..., x_k-T) o fu(u_k, ..., u_k-T) o fd(d_k, ..., d_k-T) o fe(x_k, ..., x_k-T)
        y_k =  fy(x_k, ..., x_k-T)

        Option 2 - potentially learnable time delays - TO IMPLEMENT
        x_k+1 = a_1 fx_k(x_k) o ... o a_T fx_k-T(x_k-T) o b_a fu_k(u_k) o ... o b_T fu_k-T(u_k-T)
                o c_1 fd_k(d_k) o ... o c_T fd_k-T(d_k-T) o h_1 fe_k(x_k) o ... o h_T fe_k-T(x_k-T)
        y_k = j_1 fy_k(x_k) o ... o j_T fy_k-T(x_k-T)

        :param fx: (nn.Module) State transition function
        :param fy: (nn.Module) Observation function
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param fe: (nn.Module) Error term via state augmentation
        :param xou: (callable) Elementwise tensor op
        :param xod: (callable) Elementwise tensor op
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param timedelay: (int) Number of time delays
        """
        super().__init__(fx, fy, fu=fu, fd=fd, fe=fe, xou=xou, xod=xod, xoe=xoe, residual=residual, input_keys=input_keys, name=name)
        self.nx, self.nx_td, self.ny = (self.fx.out_features, self.fx.in_features, self.fy.out_features)

        in_features = self.nx_td
        in_features = in_features + self.fu.in_features if fu is not None else in_features
        in_features = in_features + self.fd.in_features if fd is not None else in_features
        self.in_features = in_features
        self.out_features = self.fy.out_features
        self.check_features()
        self.timedelay = timedelay

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        x_in, y_out, u_in_f, u_in_p, d_in_f, d_in_p = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y, FD, FU, FE = [], [], [], [], []

        if u_in_f in data and u_in_p in data:
            Utd = torch.cat([data[u_in_p][-self.timedelay:], data[u_in_f]])  # shape=(T+nsteps, bs, nu)
        if d_in_f in data and d_in_p in data:
            Dtd = torch.cat([data[d_in_p][-self.timedelay:], data[d_in_f]])  # shape=(T+nsteps, bs, nd)
        Xtd = data[x_in]                                                     # shape=(T+1, bs, nx)
        for i in range(nsteps):
            x_prev = Xtd[-1]
            x_delayed = torch.cat([Xtd[k, :, :] for k in range(Xtd.shape[0])], dim=-1)  # shape=(bs, T*nx)
            x = self.fx(x_delayed)
            if self.fu is not None:
                Utd_i = Utd[i:i + self.timedelay + 1]
                u_delayed = torch.cat([Utd_i[k, :, :] for k in range(Utd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                fu = self.fu(u_delayed)
                x = self.xou(x, fu)
                FU.append(fu)
            if self.fd is not None:
                Dtd_i = Dtd[i:i + self.timedelay + 1]
                d_delayed = torch.cat([Dtd_i[k, :, :] for k in range(Dtd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                fd = self.fd(d_delayed)
                x = self.xod(x, fd)
                FD.append(fd)
            if self.fe is not None:
                fe = self.fe(x_delayed)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            Xtd = torch.cat([Xtd, x.unsqueeze(0)])[1:]
            y = self.fy(x_delayed)
            X.append(x)
            Y.append(y)
        output = dict()
        for tensor_list, name in zip([X, Y, FU, FD, FE],
                                     ['X_pred', 'Y_pred', 'fU', 'fD', 'fE']):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        output['reg_error'] = self.reg_error()
        return output

    @staticmethod
    def keys(input_keys):
        """
        Overwrite canonical expected input keys with alternate names

        :param input_keys: (dict {str:str}) Mapping canonical expected input keys to alternate names
        :return: (list [str]) List of input keys
        """
        default_keys = {'Xtd': 'Xtd', 'Yf': 'Yf', 'Yp': 'Yp', 'Uf': 'Uf', 'Up': 'Up', 'Df': 'Df', 'Dp': 'Dp'}
        new_keys = {**default_keys, **input_keys}
        return [new_keys['Xtd'], new_keys['Yf'], new_keys['Uf'], new_keys['Up'], new_keys['Df'], new_keys['Dp']]

    def check_features(self):
        self.nx_td, self.nx, self.ny = self.fx.in_features, self.fx.out_features, self.fy.out_features
        self.nu_td = self.fu.in_features if self.fu is not None else 0
        self.nd_td = self.fd.in_features if self.fd is not None else 0
        if self.fu is not None:
            assert self.fu.out_features == self.fx.out_features, 'Dimension mismatch between input and state transition'
        if self.fd is not None:
            assert self.fd.out_features == self.fx.out_features, 'Dimension mismatch between disturbance and state transition'


class TimeDelayBlackSSM(BlackSSM):
    def __init__(self, fxud, fy, fe=None, xoe=torch.add, name='black_ssm', input_keys=dict(), timedelay=0, residual=False):
        """
        black box state space with generic unstructured time delayed system dynamics:
        x_k+1 = fxud(x_k, ..., x_k-T, u_k, ..., u_k-T, d_k, ..., d_k-T) o fe(x_k, ..., x_k-T)
        y_k =  fy(x_k, ..., x_k-T)

        :param fxud: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param fe: (nn.Module) Error term via state augmentation
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param timedelay: (int) Number of time delays
        """
        super().__init__(fxud, fy, fe=fe, xoe=xoe, name=name, input_keys=input_keys, residual=residual)
        self.in_features = self.fxud.in_features
        self.out_features = self.fy.out_features
        self.timedelay = timedelay
        self.input_keys = self.keys(input_keys)

    def forward(self, data):
        """
        """
        x_in, y_out, u_in_f, u_in_p, d_in_f, d_in_p = self.input_keys
        nsteps = data[y_out].shape[0]
        X, Y, FE = [], [], []

        print(data)
        if u_in_f in data and u_in_p in data:
            Utd = torch.cat([data[u_in_p][-self.timedelay:], data[u_in_f]])  # shape=(T+nsteps, bs, nu)
        if d_in_f in data and d_in_p in data:
            Dtd = torch.cat([data[d_in_p][-self.timedelay:], data[d_in_f]])  # shape=(T+nsteps, bs, nd)
        Xtd = data[x_in]                                                     # shape=(T+1, bs, nx)
        for i in range(nsteps):
            x_prev = Xtd[-1]
            x_delayed = torch.cat([Xtd[k, :, :] for k in range(Xtd.shape[0])], dim=-1)  # shape=(bs, T*nx)
            features_delayed = x_delayed
            if u_in_f in data and u_in_p in data:
                Utd_i = Utd[i:i + self.timedelay + 1]
                u_delayed = torch.cat([Utd_i[k, :, :] for k in range(Utd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                features_delayed = torch.cat([features_delayed, u_delayed], dim=-1)
            if d_in_f in data and d_in_p in data:
                Dtd_i = Dtd[i:i + self.timedelay + 1]
                d_delayed = torch.cat([Dtd_i[k, :, :] for k in range(Dtd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                features_delayed = torch.cat([features_delayed, d_delayed], dim=-1)
            x = self.fxud(features_delayed)
            Xtd = torch.cat([Xtd, x.unsqueeze(0)])[1:]
            if self.fe is not None:
                fe = self.fe(x_delayed)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            y = self.fy(x_delayed)
            X.append(x)
            Y.append(y)
        output = dict()
        for tensor_list, name in zip([X, Y, FE],
                                     ['X_pred', 'Y_pred', 'fE']):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        output['reg_error'] = self.reg_error()
        return output

    @staticmethod
    def keys(input_keys):
        """
        Overwrite canonical expected input keys with alternate names

        :param input_keys: (dict {str:str}) Mapping canonical expected input keys to alternate names
        :return: (list [str]) List of input keys
        """
        default_keys = {'Xtd': 'Xtd', 'Yf': 'Yf', 'Yp': 'Yp', 'Uf': 'Uf', 'Up': 'Up', 'Df': 'Df', 'Dp': 'Dp'}
        new_keys = {**default_keys, **input_keys}
        return [new_keys['Xtd'], new_keys['Yf'], new_keys['Uf'], new_keys['Up'], new_keys['Df'], new_keys['Dp']]


def _extract_dims(datadims, keys, timedelay=0):
    xkey, ykey, ukey, dkey = BlockSSM.keys(keys)
    nx = datadims[xkey][-1]
    ny = datadims[ykey][-1]
    nu = datadims[ukey][-1] if ukey in datadims else 0
    nd = datadims[dkey][-1] if dkey in datadims else 0

    nx_td = (timedelay+1)*nx
    nu_td = (timedelay+1)*nu
    nd_td = (timedelay+1)*nd

    return nx, ny, nu, nd, nx_td, nu_td, nd_td


def block_model(kind, datadims, linmap, nonlinmap, bias, n_layers=2, fe=None, fyu=None,
              activation=nn.GELU, residual=False, linargs=dict(), timedelay=0,
              xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add, name='blockmodel', input_keys=dict()):
    """
    Generate a block-structured SSM with the same structure used across fx, fy, fu, and fd.
    """
    assert kind in _bssm_kinds, \
        f"Unrecognized model kind {kind}; supported models are {_bssm_kinds}"

    nx, ny, nu, nd, nx_td, nu_td, nd_td = _extract_dims(datadims, input_keys, timedelay)
    hsizes = [nx] * n_layers

    lin = lambda ni, no: (
        linmap(ni, no, bias=bias, linargs=linargs)
    )
    nlin = lambda ni, no: (
        nonlinmap(ni, no, bias=bias, hsizes=hsizes, linear_map=linmap, nonlin=activation, linargs=linargs)
    )

    # define (non)linearity of each component according to given model type
    if kind == "blocknlin":
        fx = nlin(nx_td, nx)
        fy = lin(nx_td, ny)
        fu = nlin(nu_td, nx) if nu != 0 else None
        fd = nlin(nd_td, nx) if nd != 0 else None
    elif kind == "linear":
        fx = lin(nx_td, nx)
        fy = lin(nx_td, ny)
        fu = lin(nu_td, nx) if nu != 0 else None
        fd = lin(nd_td, nx) if nd != 0 else None
    elif kind == "hammerstein":
        fx = lin(nx_td, nx)
        fy = lin(nx_td, ny)
        fu = nlin(nu_td, nx) if nu != 0 else None
        fd = nlin(nd_td, nx) if nd != 0 else None
    elif kind == "weiner":
        fx = lin(nx_td, nx)
        fy = nlin(nx_td, ny)
        fu = lin(nu_td, nx) if nu != 0 else None
        fd = lin(nd_td, nx) if nd != 0 else None
    else:   # hw
        fx = lin(nx_td, nx)
        fy = nlin(nx_td, ny)
        fu = nlin(nu_td, nx) if nu != 0 else None
        fd = nlin(nd_td, nx) if nd != 0 else None

    fe = (
        fe(nx_td, nx, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict())
        if kind in {"blocknlin", "hammerstein", "hw"}
        else fe(nx_td, nx, bias=bias, linargs=linargs)
    ) if fe is not None else None

    fyu = (
        fyu(nu_td, ny, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict())
        if kind in {"blocknlin", "hw"}
        else fyu(nu_td, ny, bias=bias, linargs=linargs)
    ) if fyu is not None else None

    model = (
        BlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, fyu=fyu, xoyu=xoyu, xou=xou, xod=xod, xoe=xoe, name=name, input_keys=input_keys, residual=residual)
        if timedelay == 0
        else TimeDelayBlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, xou=xou, xod=xod, xoe=xoe, name=name, timedelay=timedelay, input_keys=input_keys, residual=residual)
    )

    return model


def blackbox_model(datadims, linmap, nonlinmap, bias, n_layers=2, fe=None, fyu=None,
             activation=nn.GELU, residual=False, timedelay=0, linargs=dict(),
             xoyu=torch.add, xoe=torch.add, input_keys=dict(), name='blackbox_model'):
    """
    Black box state space model.
    """
    nx, ny, _, _, nx_td, nu_td, nd_td = _extract_dims(datadims, input_keys)
    hsizes = [nx] * n_layers

    fxud = nonlinmap(nx_td + nu_td + nd_td, nx, hsizes=hsizes,
                     bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs)
    fe = nonlinmap(nx_td, nx, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=dict()) if fe is not None else None
    fyu = fyu(nu_td, ny, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=dict()) if fyu is not None else None
    fy = linmap(nx_td, ny, bias=bias, linargs=linargs)

    model = (
        BlackSSM(fxud, fy, fe=fe, fyu=fyu, xoyu=xoyu, xoe=xoe, name=name, input_keys=input_keys, residual=residual)
        if timedelay == 0
        else TimeDelayBlackSSM(fxud, fy, fe=fe, xoe=xoe, name=name, timedelay=timedelay, input_keys=input_keys, residual=residual)
    )

    return model


ssm_models_atoms = [BlockSSM, BlackSSM, TimeDelayBlockSSM, TimeDelayBlackSSM]
ssm_models_train = [block_model, blackbox_model]
_bssm_kinds = {
    "linear",
    "hammerstein",
    "wiener",
    "hw",
    "blocknlin"
}


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    U = torch.rand(N, samples, nu)
    D = torch.rand(N, samples, nd)
    Y = torch.rand(N, samples, ny)

    data = {'x0': x, 'Uf': U, 'Df': D, 'Yf': Y}
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

    data = {'x0_new': x, 'Uf': U, 'Df': D, 'Yf_fresh': Y}
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

    # time delayed block SSM
    T = N-1   # admissible values: [0, nsteps-1]
    nx_td = (T+1)*nx
    nu_td = (T+1)*nu
    nd_td = (T+1)*nd
    X_td = torch.rand(T+1, samples, nx)
    data = {'X': X_td, 'Uf': U, 'Up': U, 'Df': D, 'Dp': D, 'Yf_fresh': Y}
    datadims = {'X': (nx,), 'Uf': (N, nu), 'Df': (N, nd), 'Up': (N, nu), 'Dp': (N, nd), 'Yf_fresh': (N, ny)}
    fx, fu, fd = [blocks.MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx_td, nu_td, nd_td]]
    fy = blocks.MLP(nx_td, ny, hsizes=[64, 64, 64])
    model = TimeDelayBlockSSM(fx, fy, fu, fd, timedelay=T, input_keys={'Xtd': 'X', 'Yf': 'Yf_fresh'})
    output = model(data)

    # time delayed black box SSM
    insize = (T + 1) * (nx+nu+nd)
    nx_td = (T + 1) * nx
    fxud = blocks.MLP(insize, nx, hsizes=[64, 64, 64])
    fy = blocks.MLP(nx_td, ny, hsizes=[64, 64, 64])
    model = TimeDelayBlackSSM(fxud, fy, timedelay=T, input_keys={'Xtd': 'X', 'Yf': 'Yf_fresh'})
    output = model(data)
