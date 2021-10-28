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
    + :math:`x_{t+1} = f_x(x_t) \\odot f_u(u_t) \\odot f_d(d_t) \\odot f_e(x_t)`
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

from neuromancer.component import Component


class BlockSSM(Component):

    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(self, fx, fy, fu=None, fd=None, fe=None, fyu=None,
                 xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add, residual=False, name='block_ssm',
                 input_key_map={}):
        """
        Block structured system dynamics:

        :param fx: (nn.Module) State transition function
        :param fy: (nn.Module) Observation function
        :param fu: (nn.Module) Input function
        :param fd: (nn.Module) Disturbance function
        :param fe: (nn.Module) Error term via state augmentation
        :param fyu: (nn.Module) Input-Observation function
        :param xou: (callable) Elementwise tensor op
        :param xod: (callable) Elementwise tensor op
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        """
        if fu is not None:
            self.DEFAULT_INPUT_KEYS = ['Uf'] + self.DEFAULT_INPUT_KEYS
            self.DEFAULT_OUTPUT_KEYS = ['fU'] + self.DEFAULT_OUTPUT_KEYS
        if fd is not None:
            self.DEFAULT_INPUT_KEYS = ['Df'] + self.DEFAULT_INPUT_KEYS
            self.DEFAULT_OUTPUT_KEYS = ['fD'] + self.DEFAULT_OUTPUT_KEYS
        if fe is not None:
            self.DEFAULT_OUTPUT_KEYS = ['fE'] + self.DEFAULT_OUTPUT_KEYS

        super().__init__(input_key_map, name)

        self.fx, self.fy, self.fu, self.fd, self.fe, self.fyu = fx, fy, fu, fd, fe, fyu
        self.nx, self.ny, self.nu, self.nd = (
            self.fx.in_features,
            self.fy.out_features,
            self.fu.in_features if fu is not None else 0,
            self.fd.in_features if fd is not None else 0
        )

        self.check_features()
        self.residual = residual

        self.xou, self.xod, self.xoe, self.xoyu = xou, xod, xoe, xoyu

    def check_features(self):
        self.nx, self.ny = self.fx.in_features, self.fy.out_features

        assert self.fx.in_features == self.fx.out_features, \
            'State transition must have same input and output dimensions'
        assert self.fy.in_features == self.fx.out_features, \
            'Output map must have same input size as output size of state transition'
        if self.fu is not None:
            assert self.fu.out_features == self.fx.out_features, \
                'Dimension mismatch between input and state transition'
        if self.fd is not None:
            assert self.fd.out_features == self.fx.out_features, \
                'Dimension mismatch between disturbance and state transition'

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        nsteps = data[self.input_key_map['Yf']].shape[0]
        X, Y, FD, FU, FE = [], [], [], [], []
        x = data[self.input_key_map['x0']]
        for i in range(nsteps):
            x_prev = x
            x = self.fx(x)
            if self.fu is not None:
                fu = self.fu(data[self.input_key_map['Uf']][i])
                x = self.xou(x, fu)
                FU.append(fu)
            if self.fd is not None:
                fd = self.fd(data[self.input_key_map['Df']][i])
                x = self.xod(x, fd)
                FD.append(fd)
            if self.fe is not None:
                fe = self.fe(x_prev)
                x = self.xoe(x, fe)
                FE.append(fe)
            if self.residual:
                x += x_prev
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(data[self.input_key_map['Uf']][i])
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)

        output = {name: torch.stack(tensor_list) for tensor_list, name
                  in zip([X, Y, FU, FD, FE], ['X_pred', 'Y_pred', 'fU', 'fD', 'fE'])
                  if tensor_list}
        output['reg_error'] = self.reg_error()
        return output

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


class BlackSSM(Component):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(self, fx, fy, fe=None, fyu=None, xoe=torch.add, xoyu=torch.add, name='black_ssm',
                 input_key_map={}, extra_inputs=[]):
        """
        Black box state space model with unstructured system dynamics:

        :param fx: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param fyu: (nn.Module) Input-Observation function
        :param fe: (nn.Module) Error term via state augmentation
        :param xoe: (callable) Elementwise tensor op
        :param xoyu: (callable) Elementwise tensor op
        :param name: (str) Name for tracking output
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param residual: (bool) Whether to make recurrence in state space model residual
        :param extra_inputs: (list of str) Input keys to be added to canonical input.
        """
        self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS + extra_inputs
        self.extra_inputs = extra_inputs
        if fe is not None:
            self.DEFAULT_OUTPUT_KEYS = ['fE'] + self.DEFAULT_OUTPUT_KEYS

        super().__init__(input_key_map, name)

        self.fx, self.fy, self.fe, self.fyu = fx, fy, fe, fyu
        self.nx, self.ny = self.fx.out_features, self.fy.out_features
        self.xoe = xoe
        self.xoyu = xoyu

    def forward(self, data):
        """
        """
        nsteps = data[self.input_key_map['Yf']].shape[0]
        X, Y, FE = [], [], []

        x = data[self.input_key_map['x0']]
        for i in range(nsteps):
            x_prev = x
            xplus = torch.cat([x] + [data[k][i] for k in self.extra_inputs], dim=1)
            x = self.fx(xplus)
            if self.fe is not None:
                fe = self.fe(x_prev)
                x = self.xoe(x, fe)
                FE.append(fe)
            y = self.fy(x)
            if self.fyu is not None:
                fyu = self.fyu(xplus)
                y = self.xoyu(y, fyu)
            X.append(x)
            Y.append(y)

        output = {name: torch.stack(tensor_list) for tensor_list, name
                  in zip([X, Y, FE], ['X_pred', 'Y_pred', 'fE'])
                  if tensor_list}
        output['reg_error'] = self.reg_error()
        return output

    def reg_error(self):
        """

        :return: 0-dimensional torch.Tensor
        """
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])

    def check_features(self):
        self.nx, self.ny = self.fx.out_features, self.fy.out_features
        assert self.fx.out_features == self.fy.in_features, 'Output map must have same input size as output size of state transition'


class TimeDelayBlockSSM(BlockSSM):
    DEFAULT_INPUT_KEYS = ["Xtd", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(self, fx, fy, fu=None, fd=None, fe=None,
                 xou=torch.add, xod=torch.add, xoe=torch.add, residual=False, name='block_ssm',
                 input_key_map={}, timedelay=0):
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
        :param input_key_map: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param timedelay: (int) Number of time delays
        """
        if fu is not None:
            self.DEFAULT_INPUT_KEYS = ['Up'] + self.DEFAULT_INPUT_KEYS
        if fd is not None:
            self.DEFAULT_INPUT_KEYS = ['Dp'] + self.DEFAULT_INPUT_KEYS
        super().__init__(fx, fy, fu=fu, fd=fd, fe=fe, xou=xou, xod=xod, xoe=xoe, residual=residual,
                         input_key_map=input_key_map, name=name)


        self.nx, self.nx_td, self.ny = (self.fx.out_features, self.fx.in_features, self.fy.out_features)

        self.check_features()
        self.timedelay = timedelay

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        nsteps = data[self.input_key_map['Yf']].shape[0]
        X, Y, FD, FU, FE = [], [], [], [], []

        if 'Uf' in self.input_key_map and 'Up' in self.input_key_map:
            Utd = torch.cat([data[self.input_key_map['Up']][-self.timedelay:], data[self.input_key_map['Uf']]])  # shape=(T+nsteps, bs, nu)
        if 'Df' in self.input_key_map and 'Dp' in self.input_key_map:
            Dtd = torch.cat([data[self.input_key_map['Dp']][-self.timedelay:], data[self.input_key_map['Df']]])  # shape=(T+nsteps, bs, nd)
        Xtd = data[self.input_key_map['Xtd']]                                                     # shape=(T+1, bs, nx)
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

        output = {name: torch.stack(tensor_list) for tensor_list, name
                  in zip([X, Y, FU, FD, FE], ['X_pred', 'Y_pred', 'fU', 'fD', 'fE'])
                  if tensor_list}
        output['reg_error'] = self.reg_error()
        return output

    def check_features(self):
        self.nx_td, self.nx, self.ny = self.fx.in_features, self.fx.out_features, self.fy.out_features
        self.nu_td = self.fu.in_features if self.fu is not None else 0
        self.nd_td = self.fd.in_features if self.fd is not None else 0
        if self.fu is not None:
            assert self.fu.out_features == self.fx.out_features, 'Dimension mismatch between input and state transition'
        if self.fd is not None:
            assert self.fd.out_features == self.fx.out_features, 'Dimension mismatch between disturbance and state transition'


class TimeDelayBlackSSM(BlackSSM):
    DEFAULT_INPUT_KEYS = ["Xtd", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(self, fx, fy, fe=None, xoe=torch.add, name='black_ssm', input_key_map={}, timedelay=0, extra_inputs=[]):
        """
        black box state space with generic unstructured time delayed system dynamics:
        x_k+1 = fx(x_k, ..., x_k-T, u_k, ..., u_k-T, d_k, ..., d_k-T) o fe(x_k, ..., x_k-T)
        y_k =  fy(x_k, ..., x_k-T)

        :param fx: (nn.Module) State transition function depending on previous state, inputs and disturbances
        :param fy: (nn.Module) Observation function
        :param fe: (nn.Module) Error term via state augmentation
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        :param timedelay: (int) Number of time delays
        :param extra_inputs: (list of str) Input keys to be added to canonical input.

        """
        super().__init__(fx, fy, fe=fe, xoe=xoe, name=name, input_key_map=input_key_map, extra_inputs=extra_inputs)
        self.timedelay = timedelay

    def forward(self, data):
        """
        """
        nsteps = data[self.input_key_map['Yf']].shape[0]
        X, Y, FE = [], [], []

        if 'Uf' in self.input_key_map and 'Up' in self.input_key_map:
            Utd = torch.cat([data[self.input_key_map['Up']][-self.timedelay:], data[self.input_key_map['Uf']]])  # shape=(T+nsteps, bs, nu)
        if 'Df' in self.input_key_map and 'Dp' in self.input_key_map:
            Dtd = torch.cat([data[self.input_key_map['Dp']][-self.timedelay:], data[self.input_key_map['Df']]])  # shape=(T+nsteps, bs, nd)
        Xtd = data[self.input_key_map['Xtd']]                                                     # shape=(T+1, bs, nx)
        for i in range(nsteps):
            x_prev = Xtd[-1]
            x_delayed = torch.cat([Xtd[k, :, :] for k in range(Xtd.shape[0])], dim=-1)  # shape=(bs, T*nx)
            features_delayed = x_delayed
            if 'Uf' in self.input_key_map and 'Up' in self.input_key_map:
                Utd_i = Utd[i:i + self.timedelay + 1]
                u_delayed = torch.cat([Utd_i[k, :, :] for k in range(Utd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                features_delayed = torch.cat([features_delayed, u_delayed], dim=-1)
            if 'Df' in self.input_key_map and 'Dp' in self.input_key_map:
                Dtd_i = Dtd[i:i + self.timedelay + 1]
                d_delayed = torch.cat([Dtd_i[k, :, :] for k in range(Dtd_i.shape[0])], dim=-1)  # shape=(bs, T*nu)
                features_delayed = torch.cat([features_delayed, d_delayed], dim=-1)
            x = self.fx(features_delayed)
            Xtd = torch.cat([Xtd, x.unsqueeze(0)])[1:]
            if self.fe is not None:
                fe = self.fe(x_delayed)
                x = self.xoe(x, fe)
                FE.append(fe)
            y = self.fy(x_delayed)
            X.append(x)
            Y.append(y)

        output = {name: torch.stack(tensor_list) for tensor_list, name
                  in zip([X, Y, FE], ['X_pred', 'Y_pred', 'fE'])
                  if tensor_list}
        output['reg_error'] = self.reg_error()
        return output


def _extract_dims(datadims, timedelay=0):
    xkey, ykey, ukey, dkey = ["x0", "Yf", "Uf", "Df"]
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
              xou=torch.add, xod=torch.add, xoe=torch.add, xoyu=torch.add, name='blockmodel', input_key_map={}):
    """
    Generate a block-structured SSM with the same structure used across fx, fy, fu, and fd.
    """
    assert kind in _bssm_kinds, \
        f"Unrecognized model kind {kind}; supported models are {_bssm_kinds}"

    nx, ny, nu, nd, nx_td, nu_td, nd_td = _extract_dims(datadims, timedelay)
    hsizes = [nx] * n_layers

    lin = lambda ni, no: (
        linmap(ni, no, bias=bias, linargs=linargs)
    )
    nlin = lambda ni, no: (
        nonlinmap(ni, no, bias=bias, hsizes=hsizes, linear_map=linmap, nonlin=activation, linargs=linargs)
    )

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
        BlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, fyu=fyu, xoyu=xoyu, xou=xou, xod=xod, xoe=xoe, name=name, input_key_map=input_key_map, residual=residual)
        if timedelay == 0
        else TimeDelayBlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, xou=xou, xod=xod, xoe=xoe, name=name, timedelay=timedelay, input_key_map=input_key_map, residual=residual)
    )

    return model


def blackbox_model(datadims, linmap, nonlinmap, bias, n_layers=2, fe=None, fyu=None,
             activation=nn.GELU, timedelay=0, linargs=dict(),
             xoyu=torch.add, xoe=torch.add, input_key_map={}, name='blackbox_model', extra_inputs=[]):
    """
    Black box state space model.
    """
    nx, ny, _, _, nx_td, nu_td, nd_td = _extract_dims(datadims)
    hsizes = [nx] * n_layers

    fx = nonlinmap(nx_td + nu_td + nd_td, nx, hsizes=hsizes,
                     bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs)
    fe = nonlinmap(nx_td, nx, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs) if fe is not None else None
    fyu = fyu(nu_td, ny, hsizes=hsizes,
                   bias=bias, linear_map=linmap, nonlin=activation, linargs=linargs) if fyu is not None else None
    fy = linmap(nx_td, ny, bias=bias, linargs=linargs)

    model = (
        BlackSSM(fx, fy, fe=fe, fyu=fyu, xoyu=xoyu, xoe=xoe, name=name,
                 input_key_map=input_key_map, extra_inputs=extra_inputs)
        if timedelay == 0
        else TimeDelayBlackSSM(fx, fy, fe=fe, xoe=xoe, name=name, timedelay=timedelay,
                               input_key_map=input_key_map, extra_inputs=extra_inputs)
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
