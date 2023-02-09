import torch
import torch.nn as nn

from neuromancer.integrators import Euler
from neuromancer.component import Component
from neuromancer.blocks import MLP


class SSMIntegrator(Component):
    """
    Component state space model wrapper for integrator
    """

    def __init__(self, integrator, encoder=torch.nn.Identity(), decoder=torch.nn.Identity()):
        """
        :param integrator: (neuromancer.integrators.Integrator)
        :param nsteps: (int) Number of rollout steps from initial condition
        """
        super().__init__(['X', 'U'], ['X_auto', 'X_step', 'X_nstep', 'Z', 'Z_step', 'Z_nstep'],
                         name='ssm')
        self.integrator = integrator
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        """
        :param data: (dict {str: Tensor}) {'U': shape=(nsamples, nsteps, nu), 'X': shape=(nsamples, nsteps, nx)}

        """
        nsteps = data['X'].shape[1]
        X = data['X']
        U = data['U']
        Z = self.encoder(X)
        X_auto = self.decoder(Z)
        Z_step = torch.cat([Z[:, 0:1, :], self.integrator(Z[:, :-1, :], u=U[:, :-1, :])], dim=1)
        X_step = self.decoder(Z_step)

        Z_nstep = [Z[:, 0:1, :]]
        z = Z[:, 0:1, :]
        for i in range(nsteps - 1):
            z = self.integrator(z, u=U[:, i:i+1, :])
            Z_nstep.append(z)
        Z_nstep = torch.cat(Z_nstep, dim=1)
        X_nstep = self.decoder(Z_nstep)
        output = {k: v for k, v in zip(['X_auto', 'X_step', 'X_nstep',
                                        'Z', 'Z_step', 'Z_nstep'],
                                       [X_auto, X_step, X_nstep,
                                        Z, Z_step, Z_nstep])}
        return output


def get_node(ny, nu, args):
    fx = MLP(ny + nu, ny, bias=False, linear_map=nn.Linear, nonlin=nn.ELU,
             hsizes=[args.hsize for h in range(args.nlayers)])
    interp_u = lambda tq, t, u: u
    integrator = Euler(fx, h=args.ts, interp_u=interp_u)
    ssm = SSMIntegrator(integrator)
    return ssm