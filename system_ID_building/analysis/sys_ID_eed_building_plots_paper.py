

# python base imports
import argparse
import dill
import warnings
import os
import scipy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# machine learning data science imports
import numpy as np
import torch
import matplotlib.patches as mpatches

try:
    from sippy import *
except ImportError:
    import sys, os
    sys.path.append(os.pardir)
    from sippy import *
from sippy import functionsetSIM as fsetSIM

# local imports
from neuromancer.plot import pltCL, pltOL, get_colors
from neuromancer.datasets import FileDataset, EmulatorDataset
import plot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=5000)
    opt_group.add_argument('-lr', type=float, default=0.003,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'],
                            default='datafile',
                            help='source type of the dataset')
    data_group.add_argument('-system', default='EED_building',
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=3000,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', choices=['UDY', 'U', 'Y', None], type=str, default='UDY')
    data_group.add_argument('-dataset_name', type=str, choices=['openloop', 'closedloop'],
                            default='openloop',
                            help='name of the dataset')
    data_group.add_argument('-model_file', type=str,
                            default='../results_files/blocknlin_constr_bias_pf_rnn_16/best_model.pth')
    data_group.add_argument('-trained_data_file', type=str,
                            default='../results_files/blocknlin_constr_bias_pf_rnn_16/best_model_data.pth')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'hw', 'hammerstein', 'blocknlin'],
                             default='blocknlin')
    model_group.add_argument('-nx_hidden', type=int, default=4, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear', 'residual_mlp'], default='mlp')
    model_group.add_argument('-nonlinear_map', type=str, default='mlp',
                             choices=['mlp', 'rnn', 'pytorch_rnn', 'linear', 'residual_mlp'])
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')


    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-verbosity', type=int, default=100,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    return parser.parse_args()


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
        self.savedir = savedir

    def eval(self, outputs):
        dsets = ['train', 'dev', 'test']
        Ypred = [unbatch_data(outputs[f'nstep_{dset}_Y_pred']).reshape(-1, self.dataset.dims['Yf'][-1]).detach().cpu().numpy() for dset in dsets]
        Ytrue = [unbatch_data(outputs[f'nstep_{dset}_Yf']).reshape(-1, self.dataset.dims['Yf'][-1]).detach().cpu().numpy() for dset in dsets]
        pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(self.savedir, 'nstep_OL.png'))

        Ypred = [outputs[f'loop_{dset}_Y_pred'].reshape(-1, self.dataset.dims['Yf'][-1]).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, self.dataset.dims['Yf'][-1]).detach().cpu().numpy() for dset in dsets]
        pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(self.savedir, 'open_OL.png'))

        trajectory_static(np.concatenate(Ytrue).transpose(1, 0),
                              np.concatenate(Ypred).transpose(1, 0),
                              figname=os.path.join(self.savedir, f'open_static.png'))

        return dict()


def pltOL(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
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
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=3, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=24)
        ax[j, 0].set_xlabel('Time', fontsize=24)
        ax[j, 0].set_ylabel(notation, fontsize=24)
        ax[j, 0].tick_params(axis='x', labelsize=22)
        ax[j, 0].tick_params(axis='y', labelsize=22)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def trajectory_static(true_traj, pred_traj, figname='traj.png'):
    plt.style.use('classic')
    fig, ax = plt.subplots(len(true_traj), 1, figsize=(16, 10))
    # labels = [f'$y_{k}$' for k in range(len(true_traj))]
    labels = range(len(true_traj))
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax if len(true_traj) == 1 else ax[row]
        axe.set(xlim=(0, t1.shape[0]),
                    ylim=(min(t1.min(), t2.min()) - 0.1, max(t1.max(), t2.max()) + 0.1))
        axe.set_ylabel('$y_{' + str(label) + '}$', rotation=0, labelpad=20, fontsize=16)
        axe.plot(t1, label='True', c='r', linewidth=1.0)
        axe.plot(t2, label='Pred', c='b', linewidth=1.0)
        axe.tick_params(labelbottom=False)
        # axe.set(yticks=(0, 1),
        #         yticklabels=('-1', '33'))
        # axe.set_ylim([0, 1])
        axe.set(yticks=(0, 1),
                yticklabels=('0', '1'),
                ylim=(-0.6, 1.6))
        axe.set(xticks=np.arange(0, t1.shape[0], step=96),
                xticklabels=(np.arange(0, t1.shape[0], step=96)/96).astype(int))
        axe.tick_params(axis='x', labelsize=10)
        axe.tick_params(axis='y', labelsize=10)
        # axe.axvspan(0, 100, facecolor='grey', alpha=0.4)
        # axe.axvspan(0, np.floor(t1.shape[0]/3), facecolor='grey', alpha=0.4, zorder=-100)
        # axe.axvspan(np.floor(t1.shape[0]/3), 2*np.floor(t1.shape[0]/3), facecolor='grey', alpha=0.2, zorder=-100)
        axe.axvspan(0, 960, facecolor='grey', alpha=0.4, zorder=-100)
        axe.axvspan(960, 2*960, facecolor='grey', alpha=0.2, zorder=-100)
    axe.tick_params(labelbottom=True)
    axe.set_xlabel('time [days]', fontsize=12)
    # axe.legend(fontsize=24)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    plt.show()
    plt.savefig(figname)

def plot_eigenvalues_compact(model, savedir='./test/'):
    Mat = []
    if hasattr(model, 'fx'):
        if hasattr(model.fx, 'effective_W'):
            rows = 1
            Mat.append(model.fx.effective_W().detach().cpu().numpy())
        elif hasattr(model.fx, 'linear'):
            rows = len(model.fx.linear)
            for linear in model.fx.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        elif hasattr(model.fx, 'rnn'):
            if isinstance(model.fx.rnn, torch.nn.RNN):
                rows = len(model.fx.rnn.all_weights[0])
                for cell in model.fx.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(model.fx.rnn.rnn_cells)
                for cell in model.fx.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
        else:
            rows = 0
    elif hasattr(model, 'fxud'):
        if hasattr(model.fxud, 'effective_W'):
            rows = 1
            Mat.append(model.fxud.effective_W().detach().cpu().numpy())
        elif hasattr(model.fxud, 'linear'):
            rows = len(model.fxud.linear)
            for linear in model.fxud.linear:
                Mat.append(linear.weight.detach().cpu().numpy())
        elif hasattr(model.fxud, 'rnn'):
            if isinstance(model.fxud.rnn, torch.nn.RNN):
                rows = len(model.fxud.rnn.all_weights[0])
                for cell in model.fxud.rnn.all_weights[0]:
                    Mat.append(cell.detach().cpu().numpy())
            else:
                rows = len(model.fxud.rnn.rnn_cells)
                for cell in model.fxud.rnn.rnn_cells:
                    Mat.append(cell.lin_hidden.effective_W().detach().cpu().numpy())
        else:
            rows = 0
    plt.style.use('classic')
    # plt.style.use(['classic', 'ggplot'])
    W = []
    Condition_numbers = []
    for k in range(rows):
        if not Mat[k].shape[0] == Mat[k].shape[1]:
            # singular values of rectangular matrix
            s, w, d = np.linalg.svd(Mat[k].T)
            w = np.sqrt(w)
        else:
            w, v = LA.eig(Mat[k].T)
        W = np.append(W, w)
        Condition_numbers = np.append(Condition_numbers, np.max(np.abs(w))/np.min(np.abs(w)))
    # https: // en.wikipedia.org / wiki / Condition_number
    print('Condition numbers are: {}'.format(Condition_numbers))

    fig, (eigax1) = plt.subplots(1, 1)
    eigax1.set_ylim(-1.1, 1.1)
    eigax1.set_xlim(-1.1, 1.1)
    eigax1.set_aspect(1)
    eigax1.scatter(W.real, W.imag, s=100, alpha=0.5, c=get_colors(len(W.real)))
    eigax1.set_xlabel('Re', fontsize=20)
    eigax1.set_ylabel('Im', fontsize=20)
    eigax1.tick_params(axis='x', labelsize=18)
    eigax1.tick_params(axis='y', labelsize=18)
    eigax1.grid(True)
    patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
    eigax1.add_patch(patch)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'eigmat.png'))


def plot_sysID(args, Y_id, Y, U=None):
    plot.pltOL(Y=Y.T, Ytrain=Y_id.T,
               figname=os.path.join(args.savedir, 'sys_ID_open_OL.png'))

    plot.trajectory_static(Y, Y_id,
                           figname=os.path.join(args.savedir, f'sys_ID_open_static.png'))


def dataset_load(args, device):
    if args.system_data == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim,
                              norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    return dataset


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

def unbatch_data(data):
    """
    Data put back together into original sequence.

    :param data: (torch.Tensor or np.array, shape=(nsteps, nsamples, dim)
    :return:  (torch.Tensor, shape=(nsim, 1, dim)
    """
    if isinstance(data, torch.Tensor):
        return data.transpose(1, 0).reshape(-1, 1, data.shape[-1])
    else:
        return data.transpose(1, 0, 2).reshape(-1, 1, data.shape[-1])


if __name__ == '__main__':
    args = parse_args()
    device = 'cpu'

    # load dataset
    dataset = dataset_load(args, device)
    normalizations = {'Ymin': -1.3061713953490333, 'Ymax': 32.77003662201578,
                      'Umin': -2.1711117, 'Umax': 33.45899931,
                      'Dmin': 29.46308055, 'Dmax': 48.97325791}

    # load trained neural state space model
    model = torch.load(args.model_file,
                       pickle_module=dill, map_location=torch.device(device))
    model_dyn = model.components[1]
    model_estim = model.components[0]

    # visualizer and simulator for open-loop traces
    visualizer = VisualizerOpen(dataset, model.components[1], args.verbosity, args.savedir)
    # plot open-loop traces
    all_output = torch.load(args.trained_data_file,
                            pickle_module=dill, map_location=torch.device(device))
    visualizer.eval(all_output)
    # plot model eigenvalues of the state transtition network weights (fx)
    plot_eigenvalues_compact(model_dyn)

    ###############################
    #### CLASSICAL SYSTEM ID    ###
    ###############################

    # Training dataset
    Y_t = dataset.train_loop['Yp'][:, 0, :].T.detach().numpy()
    U_train = dataset.train_loop['Up'][:, 0, :].T.detach().numpy()
    D_train = dataset.train_loop['Dp'][:, 0, :].T.detach().numpy()
    U_t = np.concatenate([U_train, D_train])
    length = dataset.train_loop['Yf'].shape[0]+dataset.dev_loop['Yf'].shape[0]+dataset.test_loop['Yf'].shape[0]

    # Simulation datasets
    Y = np.concatenate([dataset.train_loop['Yf'][:,0,:],
                    dataset.dev_loop['Yf'][:,0,:],
                    dataset.test_loop['Yf'][:,0,:]]).T
    U_sim = np.concatenate([dataset.train_loop['Uf'][:,0,:],
                    dataset.dev_loop['Uf'][:,0,:],
                    dataset.test_loop['Uf'][:,0,:]]).T
    D_sim = np.concatenate([dataset.train_loop['Df'][:,0,:],
                    dataset.dev_loop['Df'][:,0,:],
                    dataset.test_loop['Df'][:,0,:]]).T
    U = np.concatenate([U_sim, D_sim])
    U_sim_train = np.concatenate([dataset.train_loop['Uf'][:,0,:].T, dataset.train_loop['Df'][:,0,:].T])
    U_sim_dev = np.concatenate([dataset.dev_loop['Uf'][:,0,:].T, dataset.dev_loop['Df'][:,0,:].T])
    U_sim_test = np.concatenate([dataset.test_loop['Uf'][:,0,:].T, dataset.test_loop['Df'][:,0,:].T])


    ### Classical System identification - single run setup
    method = 'N4SID'
    nx = args.nx_hidden*dataset.dims['Y'][-1]
    SS_f = args.nsteps
    SS_p = args.nsteps
    sys_id = system_identification(Y_t, U_t, method, SS_f=SS_f, SS_p=SS_p,
                                   SS_fixed_order=nx)
    xid, yid = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0)
    xid_train, yid_train = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U_sim_train, sys_id.x0)
    xid_dev, yid_dev = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U_sim_dev, sys_id.x0)
    xid_test, yid_test = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U_sim_test, sys_id.x0)
    Yid = np.concatenate([yid_train.T, yid_dev.T, yid_test.T]).T
    # plot_sysID(args, Yid, Y, U)

    dev_MSE = ((yid_dev-dataset.dev_loop['Yf'][:, 0, :].T.detach().numpy())**2).mean()
    test_MSE = ((yid_test-dataset.test_loop['Yf'][:, 0, :].T.detach().numpy())**2).mean()

    ### System identification -  loop setup
    # methods = ['N4SID', 'MOESP', 'CVA', 'PARSIM-P', 'PARSIM-K', 'PARSIM-S']
    methods = ['N4SID', 'MOESP', 'CVA']
    nx_set = [20, 40, 60, 80]
    stability = [True, False]
    MSE_lin = dict()
    best_MSE = 1e3
    for method in methods:
        MSE_lin[method] = dict()
        for nx in nx_set:
            for stable in stability:
                sys_id = system_identification(Y_t, U_t, method, SS_f=nx, SS_p=nx,
                                               SS_fixed_order=nx, SS_A_stability=stable)
                xid, yid = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0)
                xid_train, yid_train = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D,
                                                                    U_sim_train,sys_id.x0)
                xid_dev, yid_dev = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D,
                                                                U_sim_dev, sys_id.x0)
                xid_test, yid_test = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D,
                                                                  U_sim_test, sys_id.x0)
                MSE_nstep = 0
                for k in range(U_sim_test.shape[1]//nx):
                    U_nstep_test = U_sim_test[:,k*nx:(k+1)*nx]
                    Y_nstep_test = dataset.test_loop['Yf'][k*nx:(k+1)*nx, 0, :].T.detach().numpy()
                    xid, yid = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, sys_id.C, sys_id.D,
                                                            U_nstep_test, sys_id.x0)
                    MSE_nstep += ((yid - Y_nstep_test) ** 2).mean()
                MSE_nstep = MSE_nstep/(U_sim_test.shape[1]//nx)

                dev_MSE = ((yid_dev - dataset.dev_loop['Yf'][:, 0, :].T.detach().numpy()) ** 2).mean()
                test_MSE = ((yid_test - dataset.test_loop['Yf'][:, 0, :].T.detach().numpy()) ** 2).mean()
                # MSE results
                str_sysid = str(nx)+str(stable)
                print(method+str_sysid)
                MSE_lin[method][str_sysid] = dict()
                MSE_lin[method][str_sysid]['dev'] = dev_MSE
                MSE_lin[method][str_sysid]['test'] = test_MSE
                MSE_lin[method][str_sysid]['test_nstep'] = MSE_nstep

                # save best performing trajectories
                if dev_MSE < best_MSE:
                    best_MSE = dev_MSE
                    Yid = np.concatenate([yid_train.T, yid_dev.T, yid_test.T]).T

    plot_sysID(args, Yid, Y, U)

    MSE_lin_best = dict()
    for method in methods:
        key_min_dev = min(MSE_lin[method].keys(), key=(lambda k: MSE_lin[method][k]['dev']))
        print(f'Minimum Dev {method+key_min_dev}: ', MSE_lin[method][key_min_dev]['dev'])
        key_min_test = min(MSE_lin[method].keys(), key=(lambda k: MSE_lin[method][k]['test']))
        print(f'Minimum Test {method+key_min_dev}: ', MSE_lin[method][key_min_dev]['test'])
        MSE_lin_best[method] = {}
        MSE_lin_best[method]['dev'] = MSE_lin[method][key_min_dev]['dev']
        MSE_lin_best[method]['test'] = MSE_lin[method][key_min_dev]['test']
        MSE_lin_best[method]['test_nstep'] = MSE_lin[method][key_min_dev]['test_nstep']
        MSE_lin_best[method]['test_open_K'] = min_max_denorm(MSE_lin_best[method]['test'], normalizations['Ymin'], normalizations['Ymax'])
        MSE_lin_best[method]['test_nstep_K'] = min_max_denorm(MSE_lin_best[method]['test_nstep'], normalizations['Ymin'], normalizations['Ymax'])
        print(f'Minimum open Test K {method+key_min_dev}: ', MSE_lin_best[method]['test_open_K'])
        print(f'Minimum nstep Test K {method+key_min_dev}: ', MSE_lin_best[method]['test_nstep_K'])


