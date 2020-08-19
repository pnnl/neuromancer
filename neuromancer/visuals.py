"""

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

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

    def __init__(self, dataset, model, verbosity, savedir, training_visuals=False, trace_movie=False):
        """

        :param dataset:
        :param model:
        :param verbosity:
        :param savedir:
        :param training_visuals:
        """
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        if training_visuals:
            self.anime = plot.Animator(model)
        self.training_visuals = training_visuals
        self.savedir = savedir
        self.trace_movie = trace_movie

    def train_plot(self, outputs, epoch):
        """

        :param outputs:
        :param epoch:
        :return:
        """
        if self.training_visuals:
            if epoch % self.verbosity == 0:
                self.anime()

    def plot_matrix(self):

        plt.style.use('dark_background')
        fig, (eigax, matax) = plt.subplots(1, 2)
        eigax.set_title('State Transition Matrix Eigenvalues')
        eigax.set_ylim(-1.1, 1.1)
        eigax.set_xlim(-1.1, 1.1)
        eigax.set_aspect(1)
        matax.axis('off')
        matax.set_title('State Transition Matrix')
        # TODO: treat the case when fx is not a simple linear map
        # visualize weights of modules: dict(model.fx.named_modules())
        mat = self.model.fx.effective_W().detach().cpu().numpy()
        w, v = LA.eig(mat)
        matax.imshow(mat)
        eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))
        plt.savefig(os.path.join(self.savedir, 'eigmat.png'))

    def plot_traj(self, true_traj, pred_traj, figname='open_loop.png'):
        plt.style.use('dark_background')
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

    def train_output(self):
        """

        :return:
        """
        if self.training_visuals:
            try:
                self.anime.make_and_save(os.path.join(self.savedir, 'eigen_animation.mp4'))
            except ValueError:
                pass
        return dict()

    def eval(self, outputs):
        """

        :param outputs:
        :return:
        """
        dsets = ['train', 'dev', 'test']
        ny = self.dataset.dims['Yf'][-1]
        Ypred = [unbatch_data(outputs[f'nstep_{dset}_Y_pred_dynamics']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [unbatch_data(outputs[f'nstep_{dset}_Yf']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'nstep_loop.png'))

        Ypred = [outputs[f'loop_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'open_loop.png'))
        self.plot_matrix()

        if self.trace_movie:
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

    def __init__(self, dataset, model, plot_keys, verbosity, savedir='test_control'):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys
        self.savedir = savedir

    def eval(self, outputs):
        D = outputs['D'] if 'D' in outputs.keys() else None
        R = outputs['R'] if 'R' in outputs.keys() else None
        Ymin = outputs['Ymin'] if 'Ymin' in outputs.keys() else None
        Ymax = outputs['Ymax'] if 'Ymax' in outputs.keys() else None
        Umin = outputs['Umin'] if 'Umin' in outputs.keys() else None
        Umax = outputs['Umax'] if 'Umax' in outputs.keys() else None
        plot.pltCL(Y=outputs['Y'], U=outputs['U'], D=D, R=R,
                   Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax, figname=os.path.join(self.savedir, 'CL_control.png'))
        return dict()

