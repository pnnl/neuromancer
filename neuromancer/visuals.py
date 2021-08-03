"""

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import matplotlib.patches as mpatches

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

    def __init__(self, model, verbosity, savedir, training_visuals=False, trace_movie=False):
        """

        :param model:
        :param verbosity:
        :param savedir:
        :param training_visuals:
        """
        self.model = model
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
        if hasattr(self.model, 'fx'):
            if hasattr(self.model.fx, 'effective_W'):
                rows = 1
                mat = self.model.fx.effective_W().detach().cpu().numpy()
            elif hasattr(self.model.fx, 'linear'):
                rows = len(self.model.fx.linear)
                Mat = []
                for linear in self.model.fx.linear:
                    Mat.append(linear.weight.detach().cpu().numpy())
            else:
                rows = 0
        elif hasattr(self.model, 'fxud'):
            if hasattr(self.model.fxud, 'effective_W'):
                rows = 1
                mat = self.model.fxud.effective_W().detach().cpu().numpy()
            elif hasattr(self.model.fxud, 'linear'):
                rows = len(self.model.fxud.linear)
                Mat = []
                for linear in self.model.fxud.linear:
                    Mat.append(linear.weight.detach().cpu().numpy())
            else:
                rows = 0
        else:
            rows = 0

        plt.style.use('dark_background')
        if rows == 1:
            fig, (eigax, matax) = plt.subplots(rows, 2)
            # eigax.set_ylim(-1.1, 1.1)
            # eigax.set_xlim(-1.1, 1.1)
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
            eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))
            patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
            eigax.add_patch(patch)
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))
        elif rows > 1:
            fig, axes = plt.subplots(rows, 2)
            # axes[0, 0].set_title('Weights Eigenvalues')
            axes[0, 1].set_title('State Transition Weights')
            count = 0
            for k in range(rows):
                # axes[k, 0].set_ylim(-1.1, 1.1)
                # axes[k, 0].set_xlim(-1.1, 1.1)
                axes[k, 0].set_aspect(1)
                axes[k, 1].axis('off')
                axes[k, 1].imshow(Mat[k].T)
                if not Mat[k].shape[0] == Mat[k].shape[1]:
                    # singular values of rectangular matrix
                    s, w, d = np.linalg.svd(Mat[k].T)
                    axes[k, 0].set_title('Weights Singular values')
                else:
                    w, v = LA.eig(Mat[k].T)
                    if count == 0:
                        axes[k, 0].set_title('Weights Eigenvalues')
                    count += 1
                axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[k, 0].add_patch(patch)
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))

    def plot_traj(self, true_traj, pred_traj, figname='open_loop.png'):
        try:
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
        except:
            pass

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
        ny = outputs["nstep_train_Y_pred_dynamics"].shape[-1]
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

    def __init__(self, model, plot_keys, verbosity):
        self.model = model
        self.verbosity = verbosity
        self.plot_keys = plot_keys

    def eval(self, outputs):
        data = {k:  unbatch_data(v).squeeze(1).detach().cpu().numpy()
                for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        for k, v in data.items():
            plot.plot_traj({k: v}, figname=None)
        return dict()


class VisualizerClosedLoop(Visualizer):

    def __init__(self, policy, plot_keys, ctrl_outputs, verbosity, savedir='test_control'):
        self.model = policy
        self.verbosity = verbosity
        self.plot_keys = plot_keys
        self.ctrl_outputs = ctrl_outputs
        self.savedir = savedir

    def plot_matrix(self):
        if hasattr(self.model, 'net'):
            if hasattr(self.model.net, 'effective_W'):
                rows = 1
                mat = self.model.net.effective_W().detach().cpu().numpy()
            elif hasattr(self.model.net, 'linear'):
                rows = len(self.model.net.linear)
                Mat = []
                for linear in self.model.net.linear:
                    Mat.append(linear.weight.detach().cpu().numpy())
            else:
                rows = 0
        plt.style.use('dark_background')
        if rows == 1:
            fig, (eigax, matax) = plt.subplots(rows, 2)
            # eigax.set_ylim(-1.1, 1.1)
            # eigax.set_xlim(-1.1, 1.1)
            eigax.set_aspect(1)
            matax.axis('off')
            matax.set_title('Policy Weights')
            matax.imshow(mat.T)
            if not mat.shape[0] == mat.shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(mat.T)
                eigax.set_title('Weights Singular values')
            else:
                w, v = LA.eig(mat.T)
                eigax.set_title('Weights Eigenvalues')
            eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))
        elif rows > 1:
            fig, axes = plt.subplots(rows, 2)
            # axes[0, 0].set_title('Weights Eigenvalues')
            axes[0, 1].set_title('State Transition Weights')
            count = 0
            for k in range(rows):
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
                axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))

    def eval(self, outputs):
        D = outputs.get("D", None)
        R = outputs.get("R", None)
        Ymin = outputs.get("Ymin", None)
        Ymax = outputs.get("Ymax", None)
        Umin = outputs.get("Umin", None)
        Umax = outputs.get("Umax", None)

        plot.pltCL(Y=outputs['Y'], U=outputs['U'], D=D, R=R,
                   Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                   ctrl_outputs=self.ctrl_outputs,
                   figname=os.path.join(self.savedir, 'CL_control.png'))
        self.plot_matrix()
        return dict()
