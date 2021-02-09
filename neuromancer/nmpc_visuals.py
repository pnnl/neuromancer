"""

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from neuromancer.datasets import unbatch_data, unbatch_mh_data
import neuromancer.plot as plot


class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()



class VisualizerClosedLoop2(Visualizer):

    def __init__(self, dataset, policy, plot_keys, verbosity, savedir='test_control'):
        self.model = policy
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys
        self.savedir = savedir

    def plot_traj(self, true_traj, pred_traj, figname='open_loop.png'):
        fig, ax = plt.subplots(len(true_traj), 1)
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
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

    def eval(self, outputs):
        """

        :param outputs:
        :return:
        """
        for k in ['train_model_', 'train_plant_', 'model_', 'plant_']:
            if f'{k}Y' in outputs.keys():
                D = outputs[f'{k}D'] if f'{k}D' in outputs.keys() else None
                R = outputs[f'{k}R'] if f'{k}R' in outputs.keys() else None
                Ymin = outputs[f'{k}Ymin'] if f'{k}Ymin' in outputs.keys() else None
                Ymax = outputs[f'{k}Ymax'] if f'{k}Ymax' in outputs.keys() else None
                Umin = outputs[f'{k}Umin'] if f'{k}Umin' in outputs.keys() else None
                Umax = outputs[f'{k}Umax'] if f'{k}Umax' in outputs.keys() else None
                plot.pltCL(Y=outputs[f'{k}Y'], U=outputs[f'{k}U'], D=D, R=R,
                      Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                      ctrl_outputs=self.dataset.ctrl_outputs,
                      figname=os.path.join(self.savedir, f'CL_{k}control.png'))
        return dict()



