import matplotlib.pyplot as plt
from neuromancer.callbacks import Callback
import numpy as np


def plot_traj(true_traj, pred_traj, figname='open_loop.png'):
    true_traj, pred_traj = true_traj.transpose(1, 0), pred_traj.transpose(1, 0)
    fig, ax = plt.subplots(len(true_traj), 1)
    labels = [f'$x_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax if len(true_traj) == 1 else ax[row]
        axe.set_ylabel(label, rotation=0, labelpad=20)
        axe.plot(t1, label='True', c='c')
        axe.plot(t2, label='Pred', c='m')
        axe.tick_params(labelbottom=False)
    axe.tick_params(labelbottom=True)
    axe.set_xlabel('Time')
    axe.legend()
    plt.savefig(figname)


def truncated_mse(true, pred):
    diffsq = (true - pred) ** 2
    truncs = diffsq > 1.0
    tmse = truncs * np.ones_like(diffsq) + ~truncs * diffsq
    return tmse.mean()


class TSCallback(Callback):
    def __init__(self, validator, logdir):
        self.validator = validator
        self.logdir = logdir

    def begin_eval(self, trainer, output):
        tmse, mse, sim, real = self.validator()
        output['eval_tmse'] = tmse
        output['eval_mse'] = mse
        plot_traj(real, sim, figname=f'{self.logdir}/{self.validator.figname}steps_open.png')
        plt.close()
