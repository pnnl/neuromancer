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
from neuromancer.dataset import unbatch_tensor
import neuromancer.plot as plot
from neuromancer.plot import plot_policy, plot_policy_train, \
    cl_simulate, plot_cl, plot_cl_train, plot_loss_DPC


import neuromancer.blocks as blocks


class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()


class VisualizerOpen(Visualizer):

    def __init__(self, model, verbosity, savedir, training_visuals=False, trace_movie=False, figname=None):
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
        self.figname = figname
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
            elif isinstance(self.model.fx, blocks.Linear):
                rows = 1
                mat = self.model.fx.linear.effective_W().detach().cpu().numpy()
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
        # Ypred = [unbatch_tensor(outputs[f'nstep_{dset}_Y_pred_dynamics']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        # Ytrue = [unbatch_tensor(outputs[f'nstep_{dset}_Yf']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ypred = [outputs[f'nstep_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'nstep_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]

        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'nstep_loop.png'))

        figname = self.figname if self.figname is not None else os.path.join(self.savedir, 'open_loop.png')
        Ypred = [outputs[f'loop_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=figname)
        self.plot_matrix()

        if self.trace_movie:
            plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                  np.concatenate(Ypred).transpose(1, 0),
                                  figname=os.path.join(self.savedir, f'open_movie.mp4'),
                                  freq=self.verbosity)
        return dict()


class VisualizerUncertaintyOpen(Visualizer):
    def __init__(self, dataset, savedir, dynamics_name='dynamics'):
        self.dataset = dataset
        self.savedir = savedir
        self.dynamics_name = dynamics_name

    def plot_traj(self, true_traj, pred_traj, pred_mean, pred_std, figname='open_loop.png'):
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(len(true_traj), 1)
            labels = [f'$y_{k}$' for k in range(len(true_traj))]
            for row, (t1, t2, mean, std, label) in enumerate(zip(true_traj, pred_traj, pred_mean, pred_std, labels)):
                axe = ax if len(true_traj) == 1 else ax[row]
                x = np.array(range(len(mean)))
                axe.set_ylabel(label, rotation=0, labelpad=20)
                axe.plot(t1, label='True', c='c')
                axe.fill_between(x, mean - std, mean + std, color='C3', alpha=0.2, linewidth=0)
                axe.fill_between(x, mean - std * 2, mean + std * 2, color='C3', alpha=0.2, linewidth=0)
                axe.plot(mean, label='Mean', c='C3')
                axe.tick_params(labelbottom=False)
            axe.tick_params(labelbottom=True)
            axe.set_xlabel('Time')
            axe.legend()
        except Exception as e:
            print(e)

        plt.savefig(figname)

    def eval(self, outputs):
        """
        :param outputs:
        :return:
        """
        dsets = ['train', 'dev', 'test']
        ny = self.dataset.dims['Yf'][-1]

        Ypred = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}'].reshape(-1, ny).detach().cpu().numpy() for dset in
                 dsets]
        Ymean = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}_mean'].reshape(-1, ny).detach().cpu().numpy() for
                 dset in dsets]
        Ystd = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}_std'].reshape(-1, ny).detach().cpu().numpy() for dset
                in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       np.concatenate(Ymean).transpose(1, 0),
                       np.concatenate(Ystd).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'open_loop.png'))
        return dict()


class VisualizerTrajectories(Visualizer):

    def __init__(self, model, plot_keys, verbosity):
        self.model = model
        self.verbosity = verbosity
        self.plot_keys = plot_keys

    def eval(self, outputs):
        data = {k:  unbatch_tensor(v).squeeze(1).detach().cpu().numpy()
                for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        for k, v in data.items():
            plot.plot_traj({k: v}, figname=None)
        return dict()


class VisualizerClosedLoop(Visualizer):
    def __init__(self, u_key='U_pred_policy', y_key='Y_pred_dynamics',
                 r_key='Rf', d_key=None, ymin_key=None, ymax_key=None,
                 umin_key=None, umax_key=None, policy=None,
                 ctrl_outputs=None, savedir='test_control'):
        """

        :param u_key:
        :param y_key:
        :param r_key:
        :param d_key:
        :param ymin_key:
        :param ymax_key:
        :param umin_key:
        :param umax_key:
        :param policy:
        :param ctrl_outputs:
        :param savedir:
        """
        self.model = policy
        self.u_key = u_key
        self.y_key = y_key
        self.r_key = r_key
        self.d_key = d_key
        self.ymin_key = ymin_key
        self.ymax_key = ymax_key
        self.umin_key = umin_key
        self.umax_key = umax_key
        self.ctrl_outputs = ctrl_outputs
        self.savedir = savedir

    def plot_matrix(self):
        if self.model is not None:
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
            # plt.style.use('dark_background')
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

    def eval(self, outputs, plot_weights=False, figname='CL_control.png'):

        Y = outputs[self.y_key].detach().numpy() if self.y_key is not None else None
        U = outputs[self.u_key].detach().numpy() if self.u_key is not None else None
        D = outputs[self.d_key].detach().numpy() if self.d_key is not None else None
        R = outputs[self.r_key].detach().numpy() if self.r_key is not None else None
        Ymin = outputs[self.ymin_key].detach().numpy() if self.ymin_key is not None else None
        Ymax = outputs[self.ymax_key].detach().numpy() if self.ymax_key is not None else None
        Umin = outputs[self.umin_key].detach().numpy() if self.umin_key is not None else None
        Umax = outputs[self.umax_key].detach().numpy() if self.umax_key is not None else None
        ctrl_outputs = None if self.ctrl_outputs is None else self.ctrl_outputs

        plot.pltCL(Y=Y, U=U, D=D, R=R,
                   Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                   ctrl_outputs=ctrl_outputs,
                   figname=os.path.join(self.savedir, 'CL_control.png'))
        if plot_weights:
            self.plot_matrix()
        return dict()


class VisualizerDobleIntegrator(Visualizer):
    """
    custom visualizer for double integrator example
    """
    def __init__(self, dataset, model, policy, dynamics, verbosity, savedir,
                 nstep=40, x0=1.5 * np.ones([2, 1]),
                 training_visuals=False, trace_movie=False):
        """

        :param dataset:
        :param model:
        :param verbosity:
        :param savedir:
        :param training_visuals:
        :param trace_movie:
        """
        self.model = model
        self.policy = policy
        self.dynamics = dynamics
        self.dataset = dataset
        self.verbosity = verbosity
        self.savedir = savedir
        self.training_visuals = training_visuals
        self.trace_movie = trace_movie
        self.nstep = nstep
        self.x0 = x0

    def train_output(self, trainer, epoch_policy):
        """
        visualize evolution of closed-loop contro and policy landscape during training
        :return:
        """
        A = self.dynamics.fx.linear.weight
        B = self.dynamics.fu.linear.weight
        if self.training_visuals:
            X_list, U_list = [], []
            policy_list = []
            for i in range(trainer.epochs):
                best_policy = epoch_policy[i]
                trainer.model.eval()
                policy = self.policy
                policy.load_state_dict(best_policy)
                X, U = cl_simulate(A, B, policy.net, nstep=self.nstep, x0=self.x0)
                X_list.append(X)
                U_list.append(U)
                policy_list.append(best_policy)
            plot_cl_train(X_list, U_list, nstep=self.nstep, save_path=self.savedir)
            plot_policy_train(A, B, policy, policy_list, save_path=self.savedir)
        return dict()

    def eval(self, trainer):

        A = self.dynamics.fx.linear.weight
        B = self.dynamics.fu.linear.weight
        policy = self.policy
        # plot closed loop trajectories
        X, U = cl_simulate(A, B, policy.net, nstep=self.nstep, x0=self.x0)
        plot_cl(X, U, nstep=self.nstep, save_path=self.savedir, trace_movie=self.trace_movie)
        # plot policy surface
        plot_policy(policy.net, save_path=self.savedir)
        # loss landscape and contraction regions
        plot_loss_DPC(trainer.model, policy, A, B, trainer.train_data, xmin=-2, xmax=2,
                  save_path=self.savedir)
        return dict()
