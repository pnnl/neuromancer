"""
HW in the loop setup

for k in range(nsim):
    y, r, xmin, xmax = measurements()
    x0 = estimator(y)
    u = policy(x0,r,d,xmin,xmax)
    send_control(u[0])

"""

# python base imports
import argparse
import dill
# machine learning data science imports
import numpy as np
import torch
# local imports
from neuromancer.plot import pltCL
import psl


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref_type', type=str, default='steps', choices=['steps', 'periodic'],
                        help="shape of the reference signal")
    return parser


class Simulator:
    def __init__(self, model=torch.load('../datasets/Flexy_air/best_model_flexy1.pth', pickle_module=dill)):
        self.model = model
        self.estim = model.components[0]
        self.dynamics = model.components[1]
        self.y = torch.zeros(1, 1, 1)

    def send_control(self, u, d, Y):
        estim_out = self.estim({'Yp': Y})
        inputs = {'x0_estim': estim_out['x0_estim'], 'Uf': u, 'Df': d,'Yf': Y[-1:]}
        outputs = self.dynamics(inputs)
        self.y = outputs['Y_pred_dynamics']

    def get_state(self):
        return self.y


if __name__ == '__main__':
    # trained model with estimator
    args = parse().parse_args()
    device_simulator = torch.load('../datasets/Flexy_air/best_model_flexy1.pth', pickle_module=dill)
    # trained MPC policy with model and estimator
    policy_problem = torch.load('../datasets/Flexy_air/best_policy_flexy2.pth', pickle_module=dill)
    estimator = policy_problem.components[0]
    policy = policy_problem.components[1]
    HW_emulator = Simulator(model=device_simulator)

    # dataset
    nsim = 1000
    ny = 1
    nu = 1
    nsteps = policy.nsteps

    R = {'steps': psl.Steps(nx=1, nsim=nsim, randsteps=30, xmax=0.7, xmin=0.3),
         'periodic': psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.7, xmin=0.3)}[args.ref_type]
    new_sequences = {'Y_max': 0.8 * np.ones([nsim, ny]), 'Y_min': 0.2 * np.ones([nsim, ny]),
                     'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                     'D': np.ones([nsim, 1]), 'R': R}

    yN = torch.zeros(nsteps, 1, 1)
    Y, U, R = [], [], []
    for k in range(nsim-nsteps):
        y = HW_emulator.get_state()
        yN = torch.cat([yN, y])[1:]
        estim_out = estimator({'Y_ctrl_p': yN})
        features = {'x0_estim': estim_out['x0_estim'],
                    'Rf': torch.tensor(new_sequences['R'][k:nsteps+k]).float().reshape(nsteps,1,-1),
                    'Df': torch.tensor(new_sequences['D'][k:nsteps+k]).float().reshape(nsteps,1,-1)}
        policy_out = policy(features)
        uopt = policy_out['U_pred_policy'][0].reshape(1,1,-1).float()
        d = torch.tensor(new_sequences['D'][k]).reshape(1,1,-1).float()
        HW_emulator.send_control(uopt, d=d, Y=yN)
        U.append(uopt.detach().numpy().reshape(-1))
        Y.append(y.detach().numpy().reshape(-1))
        R.append(new_sequences['R'][k])

    pltCL(Y=np.asarray(Y), R=np.asarray(R), U=np.asarray(U))