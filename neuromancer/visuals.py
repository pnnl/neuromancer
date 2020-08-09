"""
# TODO: VisualizerMPP

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np

# local imports
from neuromancer.datasets import unbatch_data
import neuromancer.plot as plot


class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()


class VisualizerOpen(Visualizer):

    def __init__(self, dataset, model, verbosity, savedir):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.anime = plot.Animator(dataset.dev_loop['Yp'].detach().cpu().numpy(), model)
        self.savedir = savedir

    def train_plot(self, outputs, epoch):
        if epoch % self.verbosity == 0:
            self.anime(outputs['loop_dev_Y_pred_dynamics'], outputs['loop_dev_Yf'])

    def train_output(self):
        self.anime.make_and_save(os.path.join(self.savedir, 'eigen_animation.mp4'))
        return dict()

    def eval(self, outputs):
        dsets = ['train', 'dev', 'test']
        ny = self.dataset.dims['Yf'][-1]
        Ypred = [unbatch_data(outputs[f'nstep_{dset}_Y_pred_dynamics']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [unbatch_data(outputs[f'nstep_{dset}_Yf']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(self.savedir, 'nstep_OL.png'))

        Ypred = [outputs[f'loop_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(self.savedir, 'open_OL.png'))

        plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                              np.concatenate(Ypred).transpose(1, 0),
                              figname=os.path.join(self.savedir, f'open_movie.mp4'),
                              freq=self.verbosity)
        return dict()


class VisualizerTrajectories(Visualizer):

    def __init__(self, dataset, model, plot_keys, verbosity):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys

    def eval(self, outputs):
        data = {k:  unbatch_data(v).squeeze(1).detach().cpu().numpy()
                for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        for k, v in data.items():
            plot.plot_traj({k: v}, figname=None)
        return dict()


class VisualizerClosedLoop(Visualizer):

    def __init__(self, dataset, model, plot_keys, verbosity):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys

    def eval(self, outputs):
        D = outputs['D'] if 'D' in outputs.keys() else None
        R = outputs['R'] if 'R' in outputs.keys() else None
        Ymin = outputs['Ymin'] if 'Ymin' in outputs.keys() else None
        Ymax = outputs['Ymax'] if 'Ymax' in outputs.keys() else None
        Umin = outputs['Umin'] if 'Umin' in outputs.keys() else None
        Umax = outputs['Umax'] if 'Umax' in outputs.keys() else None
        plot.pltCL(Y=outputs['Y'], U=outputs['U'], D=D, R=R,
                   Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax)
        return dict()

