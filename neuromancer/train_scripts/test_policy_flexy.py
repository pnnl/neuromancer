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
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.linalg as LA
import os
from scipy import signal
import matplotlib.pyplot as plt

# machine learning data science imports
import numpy as np
import torch
import matplotlib.patches as mpatches
import time
import random

# local imports
from neuromancer.plot import pltCL, pltOL, get_colors
from neuromancer.datasets import FileDataset
import psl


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref_type', type=str, default='spline',
                        choices=['steps', 'periodic', 'ramps', 'static', 'spline'],
                        help="shape of the reference signal")
    parser.add_argument('-dynamic_constraints', type=int, default=1, choices=[0, 1])
    parser.add_argument('-test_scalability', type=int, default=0, choices=[0, 1])
    return parser


class Simulator:
    def __init__(self, estimator=None, dynamics=None):
        self.estim = estimator
        self.dynamics = dynamics
        self.y = torch.ones(1, 1, 1)
        self.x = torch.zeros(1, self. dynamics.nx)

    def send_control(self, u, d, Y, x=None):
        estim_out = self.estim({'Yp': Y})
        inputs = {'x0_estim': estim_out['x0_estim'], 'U_pred_policy': u, 'Df': d,'Yf': Y[-1:]}
        outputs = self.dynamics(inputs)
        self.y = outputs['Y_pred_dynamics']
        self.x = outputs['X_pred_dynamics']

    def get_state(self):
        return self.y, self.x


class SimulatorX:
    def __init__(self, estimator=None, dynamics=None):
        self.estim = estimator
        self.dynamics = dynamics
        self.y = torch.ones(1, 1, 1)
        self.x = torch.zeros(1, self. dynamics.nx)

    def send_control(self, u, d, Y, x):
        inputs = {'x0_estim': x, 'U_pred_policy': u, 'Df': d,'Yf': Y[-1:]}
        outputs = self.dynamics(inputs)
        self.y = outputs['Y_pred_dynamics']
        self.x = outputs['X_pred_dynamics'].squeeze(0)

    def get_state(self):
        return self.y, self.x


def pltCL_paper(Y, R=None, U=None, D=None, X=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset
    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='indianred', lw=4, linestyle='--'),
                    Line2D([0], [0], color='cornflowerblue', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and R is not None:
            notation = 'position [cm]'
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(R[:, k], '--', linewidth=3, c='indianred')
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c='cornflowerblue')
                ax[j, 0].legend(custom_lines, ['Reference', 'Output'], loc='best')
                ax[j, 0].plot(Ymin[:, k], '--', linewidth=3, c='k') if Ymin is not None else None
                ax[j, 0].plot(Ymax[:, k], '--', linewidth=3, c='k') if Ymax is not None else None
        if notation == 'U':
            for k in range(array.shape[1]):
                ax[j, 0].plot(array, linewidth=3, c='cornflowerblue')
                ax[j, 0].plot(Umin[:, k], '--', linewidth=3, c='k') if Umin is not None else None
                ax[j, 0].plot(Umax[:, k], '--', linewidth=3, c='k') if Umax is not None else None
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].set(xticks=(0, 500, 1000, 1500, 2000, 2500, 3000),
                     xticklabels=('0', '125', '250', '375', '500', '625', '750'))
        ax[j, 0].grid(True)
        # ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time [s]', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=12)
        ax[j, 0].tick_params(axis='y', labelsize=12)
    # plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def pltOL_paper(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='indianred', lw=4, linestyle='--'),
                    Line2D([0], [0], color='cornflowerblue', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            notation = 'position [cm]'
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=3, c='indianred')
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c='cornflowerblue')
                ax[j, 0].legend(custom_lines, ['True', 'Pred'], fontsize=14)
                ax[j, 0].set(xticks=(0, 2000, 4000, 6000, 8000),
                             xticklabels=('0', '500', '1000', '1500', '2000'))
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        # ax[j, 0].set_title(name, fontsize=16)
        ax[j, 0].set_xlabel('Time [s]', fontsize=16)
        ax[j, 0].set_ylabel(notation, fontsize=16)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
        ax[j, 0].axvspan(0, 3000, facecolor='grey', alpha=0.4, zorder=-100)
        ax[j, 0].axvspan(3000, 6000, facecolor='grey', alpha=0.2, zorder=-100)
        ax[j, 0].margins(0, 0.1)
        ax[j, 0].text(700, -5, '             Train                ',
                   bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': 0.0})
        ax[j, 0].text(3700, -5, '           Validation           ',
                   bbox={'edgecolor': 'none', 'facecolor': 'grey', 'alpha': 0.0})
        ax[j, 0].text(6700, -5, '              Test                ',
                   bbox={'edgecolor': 'none', 'facecolor': 'grey', 'alpha': 0.0})

    # plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def plot_eigenvalues(model, savedir='./test/'):
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            mat = model.fx.effective_W().detach().cpu().numpy()
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            Mat = []
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            mat = model.fxud.effective_W().detach().cpu().numpy()
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            Mat = []
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')

    if rows == 1:
        fig, (eigax1) = plt.subplots(rows, 1)
        eigax1.set_ylim(-1.1, 1.1)
        eigax1.set_xlim(-1.1, 1.1)
        eigax1.set_aspect(1)
        if not mat.shape[0] == mat.shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(mat.T)
            eigax1.set_title('Singular values')
        else:
            w, v = LA.eig(mat.T)
            eigax1.set_title('Eigenvalues')
        eigax1.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
        eigax1.add_patch(patch)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))
    elif rows > 1:
        rows2 = int(np.ceil(rows / 2))
        fig, axes = plt.subplots(rows2, 2)
        # axes[0, 0].set_title('Weights Eigenvalues')
        count = 0
        for k in range(rows):
            axes[int(np.floor(k / 2)), 0].set_ylim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 0].set_xlim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 0].set_aspect(1)
            axes[int(np.floor(k / 2)), 1].set_ylim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 1].set_xlim(-1.1, 1.1)
            axes[int(np.floor(k / 2)), 1].set_aspect(1)
            if not Mat[k].shape[0] == Mat[k].shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(Mat[k].T)
                # axes[int(np.floor(k / 2)), 0].set_title('Singular values')
                # axes[int(np.floor(k / 2)), 1].set_title('Singular values')
            else:
                w, v = LA.eig(Mat[k].T)
                # axes[int(np.floor(k / 2)), 0].set_title('Eigenvalues') if count == 0 else None
                # axes[int(np.floor(k / 2)), 1].set_title('Eigenvalues') if count == 0 else None
                count += 1
            if k % 2 == 0:
                axes[int(np.floor(k / 2)), 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[int(np.floor(k / 2)), 0].add_patch(patch)
            else:
                axes[int(np.floor(k / 2)), 1].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[int(np.floor(k / 2)), 1].add_patch(patch)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))


def plot_weights(model, savedir='./test/'):
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            mat = model.fx.effective_W().detach().cpu().numpy()
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            Mat = []
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            mat = model.fxud.effective_W().detach().cpu().numpy()
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            Mat = []
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        else:
            rows = 0
    # plt.style.use('dark_background')
    if rows == 1:
        fig, (eigax, matax) = plt.subplots(rows, 2)
        eigax.set_ylim(-1.1, 1.1)
        eigax.set_xlim(-1.1, 1.1)
        eigax.set_aspect(1)
        matax.axis('off')
        matax.set_title('State Transition Weights')
        matax.imshow(mat.T)
        if not mat.shape[0] == mat.shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(mat.T)
            eigax.set_title('Weights Singular values')
        else:
            w, v = LA.eig(mat.T)
            eigax.set_title('Weights Eigenvalues')
        eigax.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))
    elif rows > 1:
        fig, axes = plt.subplots(rows, 2)
        # axes[0, 0].set_title('Weights Eigenvalues')
        axes[0, 1].set_title('State Transition Weights')
        count = 0
        for k in range(rows):
            axes[k, 0].set_ylim(-1.1, 1.1)
            axes[k, 0].set_xlim(-1.1, 1.1)
            axes[k, 0].set_aspect(1)
            axes[k, 1].axis('off')
            axes[k, 1].imshow(Mat[k].T)
            if not Mat[k].shape[0] == Mat[k].shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(Mat[k].T)
                axes[k, 0].set_title('Weights Singular values')
            else:
                w, v = LA.eig(Mat[k].T)
                axes[k, 0].set_title('Weights Eigenvalues') if count == 0 else None
                count += 1
            axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'eigmat.png'))


def normalize(M, Mmin=None, Mmax=None):
        """
        :param M: (2-d np.array) Data to be normalized
        :param Mmin: (int) Optional minimum. If not provided is inferred from data.
        :param Mmax: (int) Optional maximum. If not provided is inferred from data.
        :return: (2-d np.array) Min-max normalized data
        """
        Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
        Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M_norm = (M - Mmin) / (Mmax - Mmin)
        return np.nan_to_num(M_norm), Mmin.squeeze(), Mmax.squeeze()


def min_max_denorm(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = M*(Mmax - Mmin) + Mmin
    return np.nan_to_num(M_denorm)


if __name__ == '__main__':
    # trained model with estimator
    args = parse().parse_args()

    # USE min_max_denorm and normalization values to denormalize signals
    #  y - ball position
    #  u - deepMPC control action, reference signal for PID
    #  d - fan speed, control action generated by PID
    normalizations = {'Ymin': -1.3061713953490333, 'Ymax': 32.77003662201578,
                      'Umin': -2.1711117, 'Umax': 33.45899931,
                      'Dmin': 29.46308055, 'Dmax': 48.97325791}

    model = 'model4'  # choose model, choices ['model0', 'model1', 'model2', 'model3', 'model4']

    if model == 'model0':
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
        # model4 - policy with adaptive constraints
        dynamics = torch.load('../datasets/Flexy_air/device_test_models/'+model+'/best_dynamics_flexy.pth',
                                      pickle_module=dill)
        policy = torch.load('../datasets/Flexy_air/device_test_models/'+model+'/best_policy_flexy.pth',
                                    pickle_module=dill)
        estimator = torch.load('../datasets/Flexy_air/device_test_models/' + model + '/best_estimator_flexy.pth',
                            pickle_module=dill)
        estimator.input_keys[0] = 'Yp'
        policy.input_keys[0] = 'Yp'

    # temporary fix to be compatible with latest model change
    dynamics.fyu = None

    HW_emulator = Simulator(estimator=estimator, dynamics=dynamics)

    # dataset
    nsim = 9000
    dataset = FileDataset(system='flexy_air', nsim=nsim, norm=['U', 'D', 'Y'], nsteps=estimator.nsteps)
    ny = 1
    nu = 1
    nsteps = policy.nsteps

    # mean the disturbance
    disturb_static = 0.44
    # dataset.data['D'] = np.mean(dataset.data['D']) * np.ones((dataset.data['D']).shape)
    dataset.data['D'] = disturb_static * np.ones((dataset.data['D']).shape)

    # list(np.random.random(10))
    R = {'static': 0.5*np.ones([nsim,1]),
         'spline': psl.SplineSignal(nsim=nsim, values=[0.3, 0.7, 0.6, 0.8, 0.5, 0.7, 0.4, 0.8, 0.3]).reshape(nsim, 1),
         # 'spline': psl.SplineSignal(nsim=nsim, values=[0.5, 0.6, 0.4, 0.8, 0.5, 0.6, 0.5, 0.6, 0.4]).reshape(nsim,1),
         'steps': psl.Steps(nx=1, nsim=nsim, randsteps=15, xmax=0.7, xmin=0.3),
         'periodic': psl.Periodic(nx=1, nsim=nsim, numPeriods=15, xmax=0.7, xmin=0.3),
         'ramps': psl.sawtooth(nx=1, nsim=nsim, numPeriods=15, xmax=0.7, xmin=0.3)}[args.ref_type]

    if args.dynamic_constraints:
        bounds_reference = {'Y_max': psl.Periodic(nx=ny, nsim=nsim, numPeriods=30, xmax=0.9, xmin=0.6),
                            'Y_min': psl.Periodic(nx=ny, nsim=nsim, numPeriods=25, xmax=0.4, xmin=0.1),
                            'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]), 'R': R}
    else:
        bounds_reference = {'Y_max': 0.8 * np.ones([nsim, ny]), 'Y_min': 0.2 * np.ones([nsim, ny]),
                            'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]), 'R': R}

    # Open loop
    yN = torch.zeros(nsteps, 1, 1)
    Y, U = [], []
    for k in range(nsim-nsteps):
        y, x = HW_emulator.get_state()
        yN = torch.cat([yN, y])[1:]
        d = torch.tensor(dataset.data['D'][k]).reshape(1, 1, -1).float()
        u = torch.tensor(dataset.data['U'][k]).reshape(1,1,-1).float()
        HW_emulator.send_control(u, d=d, Y=yN, x=x)
        U.append(u.detach().numpy().reshape(-1))
        Y.append(y.detach().numpy().reshape(-1))
    # pltCL(Y=np.asarray(Y), R=dataset.data['Y'][:,:1], U=np.asarray(U))

    # plot open loop response for paper
    Y_plot = min_max_denorm(np.asarray(Y), normalizations['Ymin'], normalizations['Ymax'])
    R_plot = min_max_denorm(dataset.data['Y'][:,:1], normalizations['Ymin'], normalizations['Ymax'])
    pltOL_paper(Y=Y_plot, Ytrain=R_plot)
    plot_eigenvalues(dynamics)

    # Closed loop
    yN = torch.zeros(nsteps, 1, 1)
    Y, U, R = [], [], []
    Ymin, Ymax, Umin, Umax = [], [], [], []
    for k in range(nsim-nsteps):
        y, x = HW_emulator.get_state()
        yN = torch.cat([yN, y])[1:]
        estim_out = estimator({'Yp': yN})
        features = {'x0_estim': estim_out['x0_estim'], 'Yp': yN,
                    'Y_minf': torch.tensor(bounds_reference['Y_min'][k:nsteps + k]).float().reshape(nsteps, 1, -1),
                    'Y_maxf': torch.tensor(bounds_reference['Y_max'][k:nsteps + k]).float().reshape(nsteps, 1, -1),
                    'Rf': torch.tensor(bounds_reference['R'][k:nsteps+k]).float().reshape(nsteps,1,-1),
                    'Df': torch.tensor(dataset.data['D'][k:nsteps+k]).float().reshape(nsteps,1,-1)}
        policy_out = policy(features)
        uopt = policy_out['U_pred_policy'][0].reshape(1,1,-1).float()
        d = torch.tensor(dataset.data['D'][k]).reshape(1,1,-1).float()
        HW_emulator.send_control(uopt, d=d, Y=yN, x=x)
        U.append(uopt.detach().numpy().reshape(-1))
        Y.append(y.detach().numpy().reshape(-1))
        R.append(bounds_reference['R'][k])
        Ymax.append(bounds_reference['Y_max'][k])
        Ymin.append(bounds_reference['Y_min'][k])
        Umax.append(bounds_reference['U_max'][k])
        Umin.append(bounds_reference['U_min'][k])
    Y_ctrl = np.asarray(Y)
    R_ctrl = np.asarray(R)
    U_ctrl = np.asarray(U)
    Ymin_ctrl = np.asarray(Ymin)
    Ymax_ctrl = np.asarray(Ymax)
    Umin_ctrl = np.asarray(Umin)
    Umax_ctrl = np.asarray(Umax)
    # plotsteps = nsim-nsteps
    plotsteps = 3000  # plot only test set
    # pltCL_paper(Y=Y_ctrl[-plotsteps:], R=R_ctrl[-plotsteps:], U=U_ctrl[-plotsteps:],
    #       Ymin=Ymin_ctrl[-plotsteps:], Ymax=Ymax_ctrl[-plotsteps:],
    #       Umin=Umin_ctrl[-plotsteps:], Umax=Umax_ctrl[-plotsteps:])
    pltCL_paper(Y=Y_ctrl[-plotsteps:], R=R_ctrl[-plotsteps:],
          Ymin=Ymin_ctrl[-plotsteps:], Ymax=Ymax_ctrl[-plotsteps:],
          Umin=Umin_ctrl[-plotsteps:], Umax=Umax_ctrl[-plotsteps:])



    if args.test_scalability:
        scalability_models = ['policy_flexy_N5.pth', 'policy_flexy_N7.pth',
                              'policy_flexy_N10.pth', 'policy_flexy_N12.pth',
                              'policy_flexy_N15.pth']
        CPU_mean = []
        CPU_max = []
        for model in scalability_models:
            CPU = []

            policy = torch.load('../datasets/Flexy_air/scalability/' + model,
                                pickle_module=dill)
            dynamics = torch.load('../datasets/Flexy_air/device_test_models/model4/best_dynamics_flexy.pth',
                                  pickle_module=dill)
            estimator = torch.load('../datasets/Flexy_air/device_test_models/model4/best_estimator_flexy.pth',
                                   pickle_module=dill)
            estimator.input_keys[0] = 'Yp'
            policy.input_keys[0] = 'Yp'

            # Closed loop
            yN = torch.zeros(nsteps, 1, 1)
            Y, U, R = [], [], []
            Ymin, Ymax, Umin, Umax = [], [], [], []
            for k in range(nsim - nsteps):
                y, x = HW_emulator.get_state()
                yN = torch.cat([yN, y])[1:]
                estim_out = estimator({'Yp': yN})
                features = {'x0_estim': estim_out['x0_estim'], 'Yp': yN,
                            'Y_minf': torch.tensor(bounds_reference['Y_min'][k:nsteps + k]).float().reshape(nsteps, 1,
                                                                                                            -1),
                            'Y_maxf': torch.tensor(bounds_reference['Y_max'][k:nsteps + k]).float().reshape(nsteps, 1,
                                                                                                            -1),
                            'Rf': torch.tensor(bounds_reference['R'][k:nsteps + k]).float().reshape(nsteps, 1, -1),
                            'Df': torch.tensor(dataset.data['D'][k:nsteps + k]).float().reshape(nsteps, 1, -1)}

                start_step_time = time.time()
                policy_out = policy(features)
                eval_time = time.time() - start_step_time
                CPU.append(eval_time)
                uopt = policy_out['U_pred_policy'][0].reshape(1, 1, -1).float()
                d = torch.tensor(dataset.data['D'][k]).reshape(1, 1, -1).float()
                HW_emulator.send_control(uopt, d=d, Y=yN, x=x)
                # U.append(uopt.detach().numpy().reshape(-1))
                # Y.append(y.detach().numpy().reshape(-1))
                # R.append(bounds_reference['R'][k])
                # Ymax.append(bounds_reference['Y_max'][k])
                # Ymin.append(bounds_reference['Y_min'][k])
                # Umax.append(bounds_reference['U_max'][k])
                # Umin.append(bounds_reference['U_min'][k])
            # pltCL(Y=np.asarray(Y), R=np.asarray(R), U=np.asarray(U),
            #       Ymin=np.asarray(Ymin), Ymax=np.asarray(Ymax),
            #       Umin=np.asarray(Umin), Umax=np.asarray(Umax))
            CPU_mean.append(np.mean(CPU)*1000)
            CPU_max.append(np.max(CPU)*1000)