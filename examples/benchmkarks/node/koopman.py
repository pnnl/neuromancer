import torch.nn as nn
from neuromancer.blocks import MLP
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import slim
from neuromancer.activations import SoftExponential
from neuromancer.component import Component
import math


class INN(nn.Module):
    """
    For use with a Koopman operator. The INN has the same input and output sizes so as we wish expand
    the dimensionality from the original state space the deep Koopman SSM will pad the input with
    zeros
    """

    def __init__(self, nx, nz, bias=True, linear_map=slim.Linear,
                 nonlin=SoftExponential, hsizes=[64, 64, 64], linargs=dict(), n_subnets=2):
        """
        :param nx: (int) True forward mode input size prior to padding.
        :param nz: (int) This determines the input dimension and output dimension of the INN
        :param bias: (bool) Whether to use bias on the subnets
        :param linear_map: (class) Class constructor for a slim.linear map for constructing the subnetworks
        :param nonlin: (class) Class constructor for a nn.Module activation function for constructing the subnetworks
        :param hsizes: (list of int) Hidden size dimensions for subnetworks
        :param linargs: (dict) Dictionary of arguments for custom linear maps
        :param n_subnets: (int) How many neural network layers to use between coupling blocks
        """
        super().__init__()
        self.npad = nz - nx
        self.nx = nx
        self.nz = nz
        self.inn = Ff.SequenceINN(nz)
        subnet = self._define_subnet_constructor(bias=bias, linear_map=linear_map, nonlin=nonlin,
                                                 hsizes=hsizes, linargs=linargs)
        for k in range(n_subnets):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=subnet, permute_soft=False)

    def _define_subnet_constructor(self, bias=True, linear_map=slim.Linear, nonlin=SoftExponential,
                                   hsizes=[512, 512], linargs=dict()):
        def subnet_fc(c_in, c_out):
            return MLP(c_in, c_out, hsizes=hsizes, bias=bias,
                       linear_map=linear_map, nonlin=nonlin, linargs=linargs)

        return subnet_fc

    def step(self, x, rev=False):
        if rev:
            x, jac = self.inn(x, rev=rev)
            x = x[..., -self.nx:]
        else:
            pad = torch.zeros(*x.shape[:-1], self.npad)
            x = torch.cat([x, pad], dim=-1)
            x, jac = self.inn(x, rev=rev)
        return x

    def forward(self, x, rev=False):
        """
        For forward mode padding is added to x.
        For reverse mode nonpadded indices are selected from x.
        """
        shape = x.shape
        if len(x.shape) > 2:
            x = x.view(math.prod(x.shape[:-1]), -1)
        if rev:
            x, jac = self.inn(x, rev=rev)
            x = x[..., -self.nx:]
        else:
            pad = torch.randn(*x.shape[:-1], self.npad)
            x = torch.cat([pad, x], dim=-1)
            x, jac = self.inn(x, rev=rev)
        x = x.view(*shape[:-1], x.shape[-1])
        return x


class Autoencoder(nn.Module):

    def __init__(
            self,
            nx,
            nz,
            bias=True,
            linear_map=slim.Linear,
            nonlin=SoftExponential,
            hsizes=[64, 64, 64],
            linargs=dict(),
    ):
        """
        :param nx: (int) x (state) dimension
        :param nz: (int) z (lifted) dimension
        :param bias: (bool) Whether to use bias on the subnets
        :param linear_map: (class) Class constructor for a slim.linear map for constructing the subnetworks
        :param nonlin: (class) Class constructor for a nn.Module activation function for constructing the subnetworks
        :param hsizes: (list of int) Hidden size dimensions for subnetworks
        :param linargs: (dict) Dictionary of arguments for custom linear maps
        """
        super().__init__()
        self.encoder = MLP(nx, nz, bias=bias, linear_map=linear_map,
                           nonlin=nonlin, hsizes=hsizes, linargs=linargs)
        self.decoder = MLP(nz, nx, bias=bias, linear_map=linear_map,
                           nonlin=nonlin, hsizes=hsizes, linargs=linargs)
        self.nx, self.nz = nx, nz

    def forward(self, x, rev=False):
        """
        Forward mode uses encoder to map to z.
        Rev mode uses decoder to map back to x
        """
        if rev:
            return self.decoder(x)
        else:
            return self.encoder(x)


class Autoencoder(nn.Module):

    def __init__(
            self,
            nx,
            nz,
            bias=True,
            linear_map=slim.Linear,
            nonlin=SoftExponential,
            hsizes=[64, 64, 64],
            linargs=dict(),
            map=MLP
    ):
        """
        :param nx: (int) x (state) dimension
        :param nz: (int) z (lifted) dimension
        :param bias: (bool) Whether to use bias on the subnets
        :param linear_map: (class) Class constructor for a slim.linear map for constructing the subnetworks
        :param nonlin: (class) Class constructor for a nn.Module activation function for constructing the subnetworks
        :param hsizes: (list of int) Hidden size dimensions for subnetworks
        :param linargs: (dict) Dictionary of arguments for custom linear maps
        :param map: (nn.Module) Function class for mapping from x to z and back default blocks.MLP
        """
        super().__init__()
        self.encoder = map(nx, nz, bias=bias, linear_map=linear_map,
                           nonlin=nonlin, hsizes=hsizes, linargs=linargs)
        self.decoder = map(nz, nx, bias=bias, linear_map=linear_map,
                           nonlin=nonlin, hsizes=hsizes, linargs=linargs)
        self.nx, self.nz = nx, nz

    def forward(self, x, rev=False):
        """
        Forward mode uses encoder to map to z.
        Rev mode uses decoder to map back to x
        """
        if rev:
            return self.decoder(x)
        else:
            return self.encoder(x)


class StateInclusive(nn.Module):

    def __init__(
            self,
            nx,
            nz,
            bias=True,
            linear_map=slim.Linear,
            nonlin=SoftExponential,
            hsizes=[64, 64, 64],
            linargs=dict(),
    ):
        """
        :param nx: (int) x (state) dimension
        :param nz: (int) z (lifted) dimension
        :param bias: (bool) Whether to use bias on the encoder
        :param linear_map: (class) Class constructor for a slim.linear map for encoder
        :param nonlin: (class) Class constructor for a nn.Module activation function for encoder
        :param hsizes: (list of int) Hidden size dimensions for encoder
        :param linargs: (dict) Dictionary of arguments for custom linear maps
        """
        super().__init__()
        self.nx, self.nz = nx, nz + nx
        self.encoder = MLP(nx, nz, bias=bias, linear_map=linear_map,
                           nonlin=nonlin, hsizes=hsizes, linargs=linargs)

    def forward(self, x, rev=False):
        """
        Forward mode maps x to z by concatenating x to encoder output.
        Rev mode maps z to x by slicing state variables from z.
        """

        if rev:
            x = x[..., -self.nx:]
        else:
            x = torch.cat([self.encoder(x), x], dim=-1)
        return x


class KoopmanSSM(Component):

    def __init__(self, inn, koopman, name='ssm'):
        """

        :param inn: (nn.Module) Invertible neural network forward maps x to z, reverse z to x
                                Can be either INN, StateInclusive, or Autoencoder
        :param koopman: (slim.LinearBase) Maps nz to nz
        :param name: (str) Identifier for component
        """

        super().__init__(['X'], ['X_auto', 'X_step', 'X_nstep', 'Z', 'Z_step', 'Z_nstep'],
                         name=name)
        self.inn = inn
        self.K = koopman

    def forward(self, data):
        """
        :param data: (dict: {str: Tensor}) Should contain entry with key 'X' and shape (nbatch, nsteps, nx)
        :return: output (dict: {str: Tensor}) Train so that Z = Z_step = Z_nstep and X = X_auto = X_step = X_nstep
        """
        nsteps = data['X'].shape[1]
        X = data['X']
        Z = self.inn(X)
        X_auto = self.inn(Z, rev=True)
        Z_step = torch.cat([Z[:, 0:1, :], self.K(Z[:, :-1, :])])
        X_step = self.inn(Z_step, rev=True)

        Z_nstep = [Z[:, 0:1, :]]
        z = Z[:, 0:1, :]
        for i in range(nsteps - 1):
            z = self.K(z)
            Z_nstep.append(z)
        Z_nstep = torch.cat(Z_nstep, dim=1)
        X_nstep = self.inn(Z_nstep, rev=True)
        output = {k: v for k, v in zip(['X_auto', 'X_step', 'X_nstep',
                                        'Z', 'Z_step', 'Z_nstep'],
                                       [X_auto, X_step, X_nstep,
                                        Z, Z_step, Z_nstep])}
        return output


class Fxud(nn.Module):
    """
    Helper class for handling different assumptions about control input
    """

    def __init__(self, map, input_keys=['U']):
        super().__init__()
        self.input_keys, self.map = input_keys, map

    def forward(self, data):
        return self.map(torch.cat([data[k] for k in self.input_keys], dim=-1))


class NonautoKoopmanSSM(Component):

    def __init__(self, inn, koopman, fxud, control='add', name='ssm'):
        """

        :param inn: (nn.Module) Invertible neural network forward maps x to z, reverse z to x
                                Can be either INN, StateInclusive, or Autoencoder
        :param koopman: (slim.LinearBase) Linear map for koopman operator
        :param fxud: (Fxud) Will take data and concatenate inputs (e.g. x, u, d or some subset including u) and map.
                            Map could be linear or some kind of neural network.
        :param control: (str) How koopman with control is formulated. Can be 'add': K(inn(x)) + fxud(data) or
                              'shift': K(inn(x) - fuxd(data)) + fxud(data)
        :param name: (str) For identifying component
        """
        input_keys = ['X', 'U']
        super().__init__(input_keys, ['X_auto', 'X_step', 'X_nstep', 'Z', 'Z_step', 'Z_nstep'],
                         name=name)
        self.inn = inn
        self.K = koopman
        self.control = {'add': self.add, 'shift': self.shift}[control]
        self.fxud = fxud

    def shift(self, z, fxud):
        return self.K(z - fxud) + fxud

    def add(self, z, fxud):
        return self.K(z) + fxud

    def forward(self, data):
        """
        :param data: (dict: {str: Tensor}) Should contain
        :return: output (dict: {str: Tensor}) Train so that Z = Z_step = Z_nstep and X = X_auto = X_step = X_nstep
        """
        nsteps = data['X'].shape[1]
        FXUD = self.fxud(data)
        X = data['X']
        Z = self.inn(X)
        X_auto = self.inn(Z, rev=True)

        Z_step = torch.cat([Z[:, 0:1, :], self.control(Z, FXUD)[:, :-1, :]], dim=1)
        X_step = self.inn(Z_step, rev=True)

        Z_nstep = [Z[:, 0:1, :]]
        z = Z[:, 0:1, :]
        for i in range(nsteps - 1):
            z = self.control(z, FXUD[:, i:i + 1, :])
            Z_nstep.append(z)

        Z_nstep = torch.cat(Z_nstep, axis=1)
        X_nstep = self.inn(Z_nstep, rev=True)
        output = {k: v for k, v in zip(['X_auto', 'X_step', 'X_nstep',
                                        'Z', 'Z_step', 'Z_nstep'],
                                       [X_auto, X_step, X_nstep,
                                        Z, Z_step, Z_nstep])}
        return output
