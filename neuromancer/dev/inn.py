import torch.nn as nn
from neuromancer.blocks import MLP
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import slim
from neuromancer.activations import SoftExponential
from neuromancer.component import Component


class INN(nn.Module):
    """
    For use with a Koopman operator. The INN has the same input and output sizes so as we wish expand
    the dimensionality from the original state space the deep Koopman SSM will pad the input with either
    zeros or samples from a standard normal distribution.
    """
    def __init__(self, inn_dim=10, n_subnets=2, bias=True, linear_map=slim.Linear,
                 nonlin=SoftExponential, hsizes=[512, 512], linargs=dict(), permute_soft=False):
        """

        :param inn_dim: (int) This determines the input dimension and output dimension of the INN
        :param n_subnets: (int) How many neural network layers to use between coupling blocks
        :param bias: (bool) Whether to use bias on the subnets
        :param linear_map: (class) Class constructor for a slim.linear map for constructing the subnetworks
        :param nonlin: (class) Class constructor for a nn.Module activation function for constructing the subnetworks
        :param hsizes: (list of int) Hidden size dimensions for subnetworks
        :param linargs: (dict) Dictionary of arguments for custom linear maps
        :param permute_soft: (dict) Whether to use expensive "soft" permutations.
        """
        super().__init__()
        self.in_features, self.out_features = inn_dim, inn_dim
        self.inn = Ff.SequenceINN(inn_dim)
        subnet = self._define_subnet_constructor(bias=bias, linear_map=linear_map, nonlin=nonlin,
                                                 hsizes=hsizes, linargs=linargs)
        for k in range(n_subnets):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=subnet, permute_soft=permute_soft)

    def _define_subnet_constructor(self, bias=True, linear_map=slim.Linear, nonlin=SoftExponential,
                                   hsizes=[512, 512], linargs=dict()):
        def subnet_fc(c_in, c_out):
            return MLP(c_in, c_out, hsizes=hsizes, bias=bias,
                       linear_map=linear_map, nonlin=nonlin, linargs=linargs)
        return subnet_fc

    def forward(self, x, rev=False):
        return self.inn(x, rev=rev)


class InvertibleKoopmanSSM(Component):
    DEFAULT_INPUT_KEYS = ["X"]
    DEFAULT_OUTPUT_KEYS = ['X_padd', 'X_auto', 'X_step', 'X_nstep',
                           'Z', 'Z_step', 'Z_nstep',
                           'jac', 'rjac_auto', 'rjac_step', 'rjac_nstep']

    def __init__(self, xdim, zdim, inn, pad_func=torch.zeros, name='ssm', input_key_map=dict()):
        """

        :param xdim: (int) State space dimension
        :param zdim: (int) Observable (lifted space) dimension
        :param inn: (nn.Module) Invertible neural network
        :param pad_func:
        :param name:
        :param input_key_map:
        """

        super().__init__(input_key_map, name)
        self.n_pad = xdim - zdim
        self.pad_func = pad_func
        self.inn = inn
        self.in_features, self.out_features, self.name = xdim, xdim, name
        self.K = nn.Linear(zdim, zdim, bias=False)

    def forward(self, data):
        """
        :param data: (dict: {str: Tensor}) Should contain
        :return: output (dict: {str: Tensor}) Train so that Z = Z_step = Z_nstep and X = X_auto = X_step = X_nstep
        """
        nsteps = data['X'].shape[0]
        X = data['X']
        if self.pad_func is not None:
            X_padd = torch.cat(X, self.pad_func(*X.shape[:-1], self.n_pad, device=X.device), dim=-1)
        else:
            X_padd = X
        Z, jac = self.inn(X_padd)

        X_auto, rjac_auto= self.inn(Z, rev=True)
        Z_step = torch.cat([Z[0:1], self.K(Z[:-1])])
        X_step, rjac_step = self.inn(Z_step, rev=True)


        Zf_nstep = [Z[0]]
        z = Z[0]
        for i in range(nsteps-1):
            z = self.K(z)
            Zf_nstep.append(z)
        Z_nstep = torch.stack(Zf_nstep)
        X_nstep, rjac_nstep = self.inn(Z_nstep, rev=True)
        output = {k: v for k, v in zip(['X_padd', 'X_auto', 'X_step', 'X_nstep',
                                        'Z', 'Z_step', 'Z_nstep',
                                        'jac', 'rjac_auto', 'rjac_step', 'rjac_nstep'],
                                       [X_padd, X_auto, X_step, X_nstep,
                                        Z, Z_step, Z_nstep,
                                        jac, rjac_auto, rjac_step, rjac_nstep])}
        return output


class KO_state_transition(nn.Module):
    def __init__(self, obs_dim, tot_input_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.tot_input_dim = tot_input_dim
        self.KO = nn.Linear(self.obs_dim, self.obs_dim, bias=False)
        self.B = nn.Linear(self.tot_input_dim, self.obs_dim, bias=False)
        self.in_features = obs_dim

    def forward(self, x, u, d):
        ud_in = torch.cat((u, d), dim=-1)
        Bu = self.B(ud_in)
        return self.KO(x - Bu) + Bu


class NonAutonomousInvertibleKoopmanSSM(Component):
    DEFAULT_INPUT_KEYS = ["X", "U", "D"]
    DEFAULT_OUTPUT_KEYS = ['X_padd', 'X_auto', 'X_step', 'X_nstep',
                           'Z', 'Z_step', 'Z_nstep',
                           'jac', 'rjac_auto', 'rjac_step', 'rjac_nstep']

    def __init__(self, xdim, zdim, inn, udim, ddim, pad_func=torch.zeros, name='ssm', input_key_map=dict()):
        """

        """

        super().__init__(input_key_map, name)
        assert zdim > xdim, f'Latent space dimension: {zdim} should be larger than state space dimension: {xdim}'
        assert inn.in_features == zdim
        self.n_pad = xdim - zdim
        self.pad_func = pad_func
        self.inn = inn
        self.in_features, self.out_features, self.name = xdim, xdim, name
        self.K = KO_state_transition(zdim, zdim, bias=False)

    def forward(self, data):
        """
        :param data: (dict: {str: Tensor}) Should contain
        :return: output (dict: {str: Tensor}) Train so that Z = Z_step = Z_nstep and X = X_auto = X_step = X_nstep
        """
        nsteps = data['X'].shape[0]
        X = data['X']
        U = data['U']
        D = data['D']
        X_padd = torch.cat(X, self.pad_func(*X.shape[:-1], self.n_pad, device=X.device), dim=-1)
        Z, jac = self.inn(X_padd)

        X_auto, rjac_auto = self.inn(Z, rev=True)
        Z_step = torch.cat([Z[0:1], self.K(Z[:-1], U[:-1], D[:-1])])
        X_step, rjac_step = self.inn(Z_step, rev=True)


        Zf_nstep = [Z[0]]
        z = Z[0]
        for i in range(nsteps-1):
            z = self.K(z, U[i], D[i])
            Zf_nstep.append(z)
        Z_nstep = torch.stack(Zf_nstep)
        X_nstep, rjac_nstep = self.inn(Z_nstep, rev=True)
        output = {k: v for k, v in zip(['X_padd', 'X_auto', 'X_step', 'X_nstep',
                                        'Z', 'Z_step', 'Z_nstep',
                                        'jac', 'rjac_auto', 'rjac_step', 'rjac_nstep'],
                                       [X_padd, X_auto, X_step, X_nstep,
                                        Z, Z_step, Z_nstep,
                                        jac, rjac_auto, rjac_step, rjac_nstep])}
        return output
