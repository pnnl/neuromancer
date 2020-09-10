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
from neuromancer.plot import pltCL, pltOL
from neuromancer.datasets import FileDataset
import psl


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref_type', type=str, default='steps', choices=['steps', 'periodic'],
                        help="shape of the reference signal")
    return parser


class Simulator:
    def __init__(self, estimator=None, dynamics=None):
        self.estim = estimator
        self.dynamics = dynamics
        self.y = torch.zeros(1, 1, 1)
        self.x = torch.zeros(1, 1,self. dynamics.nx)

    def send_control(self, u, d, Y, x=None):
        estim_out = self.estim({'Yp': Y})
        inputs = {'x0_estim': estim_out['x0_estim'], 'U_pred_policy': u, 'Df': d,'Yf': Y[-1:]}
        outputs = self.dynamics(inputs)
        self.y = outputs['Y_pred_dynamics']
        self.x = outputs['X_pred_dynamics']

    def get_state(self):
        return self.y, self.x


if __name__ == '__main__':
    # trained model with estimator
    args = parse().parse_args()

    model = 'model2'

    if model == 'model1':
        # state feedback policy with estimator in the loop
        device_simulator = torch.load('../datasets/Flexy_air/device_test_models/model0/best_model_flexy1.pth', pickle_module=dill)
        policy_problem = torch.load('../datasets/Flexy_air/device_test_models/model0/best_model_flexy1_policy1.pth', pickle_module=dill)
        estimator = policy_problem.components[0]
        estimator.input_keys[0] = 'Yp'
        dynamics = device_simulator.components[1]
        dynamics.input_keys[2] = 'U_pred_policy'
        policy = policy_problem.components[1]
    else:
        # output feedback policy
        dynamics = torch.load('../datasets/Flexy_air/device_test_models/'+model+'/best_dynamics_flexy.pth',
                                      pickle_module=dill)
        policy = torch.load('../datasets/Flexy_air/device_test_models/'+model+'/best_policy_flexy.pth',
                                    pickle_module=dill)
        estimator = torch.load('../datasets/Flexy_air/device_test_models/' + model + '/best_estimator_flexy.pth',
                            pickle_module=dill)
        estimator.input_keys[0] = 'Yp'
        policy.input_keys[0] = 'Yp'

    HW_emulator = Simulator(estimator=estimator, dynamics=dynamics)

    # dataset
    nsim = 3000
    dataset = FileDataset(system='flexy_air', nsim=nsim, norm=['U', 'D', 'Y'], nsteps=estimator.nsteps)
    ny = 1
    nu = 1
    nsteps = policy.nsteps

    R = {'steps': psl.Steps(nx=1, nsim=nsim, randsteps=30, xmax=0.7, xmin=0.3),
         'periodic': psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.7, xmin=0.3)}[args.ref_type]
    new_sequences = {'Y_max': 0.8 * np.ones([nsim, ny]), 'Y_min': 0.2 * np.ones([nsim, ny]),
                     'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                     'D': np.ones([nsim, 1]), 'R': R}

    # Open loop
    yN = torch.zeros(nsteps, 1, 1)
    Y, U, R = [], [], []
    for k in range(nsim-nsteps):
        y, x = HW_emulator.get_state()
        yN = torch.cat([yN, y])[1:]
        d = torch.tensor(new_sequences['D'][k]).reshape(1,1,-1).float()
        u = torch.tensor(dataset.data['U'][k]).reshape(1,1,-1).float()
        HW_emulator.send_control(u, d=d, Y=yN)
        U.append(u.detach().numpy().reshape(-1))
        Y.append(y.detach().numpy().reshape(-1))
        R.append(new_sequences['R'][k])
    pltCL(Y=np.asarray(Y), R=dataset.data['Y'][:,:1], U=np.asarray(U))

    # Closed loop
    yN = torch.zeros(nsteps, 1, 1)
    Y, U, R = [], [], []
    for k in range(nsim-nsteps):
        y, x = HW_emulator.get_state()
        yN = torch.cat([yN, y])[1:]
        estim_out = estimator({'Yp': yN})
        features = {'x0_estim': estim_out['x0_estim'], 'Yp': yN,
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