"""


TODO: Make these comments reflect current code
This script can train building dynamics and state estimation models with the following
cross-product of configurations

Dynamics:
    Ground truth, linear, Perron-Frobenius normalized linear, SVD decomposition normalized linear, Spectral linear
Heat flow:
    Black, grey, white
State estimation:
    Ground truth, linear, Perron-Frobenius linear, vanilla RNN,
    Perron-Frobenius RNN, SVD decomposition RNN, Spectral RNN, linear Kalman Filter
Ground Truth Model:
    Large full building thermal model
    Large reduced order building thermal model
Bias:
    Linear transformations
    Affine transformations
Constraints:
    Training with constraint regularization
    Training without regularization
Normalization:
    Training with input normalization
    Training with state normalization
    Training with no normalization

Several hyperparameter choices are also available and described in the argparse.

training options:
1, control via closed loop model
        trainable modules: SSM + estim + policy
        choices: fully/partially observable, w/wo measured disturbances d, SSM given or learned
2, sytem ID via open loop model
        trainable modules: SSM + estim
        choices: fully/partially observable, w/wo measured disturbances D
3, time series via open loop model
        trainable modules: SSM + estim
        no inputs U and disturbances D
"""
# python imports
import os
import argparse
from copy import deepcopy
import time
# plotting imports
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
# ml imports
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg as LA
# local imports
import plot
from linear import Linear
import dataset
import ssm
import estimators
import policies
import loops
import nlinear as linear
import blocks
import emulators
import rnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=200)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'], default='datafile')
    data_group.add_argument('-datafile', default='./datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat',
                            help='source of the dataset')
    data_group.add_argument('-norm', type=str, default='UDY')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-loop', type=str,
                             choices=['open', 'closed'], default='open')
    model_group.add_argument('-ssm_type', type=str, choices=['GT', 'BlockSSM', 'BlackSSM'], default='BlockSSM')
    model_group.add_argument('-nx_hidden', type=int, default=5, help='Number of hidden states per output')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()), default='linear')
    model_group.add_argument('-nonlinear_map', type=str,
                             choices=['mlp', 'rnn', 'linear', 'residual_mlp', 'sparse_residual_mlp'], default='residual_mlp')
    model_group.add_argument('-nonlin', type=str,
                             choices=['relu', 'gelu'], default='gelu')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_u', type=float,  default=0.0, help='Relative penalty on hidden input constraints.')
    weight_group.add_argument('-Q_con_x', type=float,  default=0.0, help='Relative penalty on hidden state constraints.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=0.0, help='Relative penalty on maximal influence of u and d on hidden state in one time step.')
    weight_group.add_argument('-Q_dx', type=float,  default=0.0, help='Relative penalty on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Relative penalty linear maps regularization.')
    weight_group.add_argument('-Q_y', type=float,  default=1e0, help='Relative penalty on output tracking.')


    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-verbosity', type=int, default=100,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', default='deepmpc',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-mlflow', action='store_true',
                           help='Using mlflow or not.')
    log_group.add_argument('-make_movie', action='store_true', help='Make movies with this flag.')
    log_group.add_argument('-freq', type=int, help='Frequency to create frames for reference tracking movie.', default=1)
    return parser.parse_args()


# single training step
def step(loop, data):
    if type(loop) is loops.OpenLoop:
        Yp, Yf, Up, Uf, Dp, Df = data
        X_pred, Y_pred, reg_error = loop(Yp, Up, Uf, Dp, Df)
        U_pred = Uf
        criterion = torch.nn.MSELoss()
        loss = criterion(Y_pred.squeeze(), Yf.squeeze())

    elif type(loop) is loops.ClosedLoop:
        Yp, Yf, Up, Dp, Df, Rf = data
        X_pred, Y_pred, U_pred, reg_error = loop(Yp, Up, Dp, Df, Rf)
        loss = None
    # TODO: extent this to two control options: w/wo given model
    # TODO: Library of custom loss functions??
    # TODO: shall we create separate file losses.py with various types of loss functions?

    return loss, reg_error, X_pred, Y_pred, U_pred


class Animator:

    def __init__(self, Y, data, loop):
        self.data = [d.transpose(0, 1).reshape(1, -1, d.shape[-1]).transpose(0, 1)
                     if d is not None else d for d in data]
        self.loop = loop
        nsteps, ny = Y.shape
        plt.style.use('dark_background')
        self.fig = plt.figure(constrained_layout=True)
        gs = GridSpec(nrows=1+ny, ncols=2, figure=self.fig, width_ratios=[1,1],
                                  height_ratios=[5] + [1]*ny)
        self.eigax = self.fig.add_subplot(gs[0, 1])
        self.eigax.set_title('State Transition Matrix Eigenvalues')
        self.eigax.set_ylim(-1.1, 1.1)
        self.eigax.set_xlim(-1.1, 1.1)
        self.eigax.set_aspect(1)

        self.matax = self.fig.add_subplot(gs[0, 0])
        self.matax.axis('off')
        self.matax.set_title('State Transition Matrix')

        self.trjax = [self.fig.add_subplot(gs[k, :]) for k in range(1, ny+1)]
        for row, ax in enumerate(self.trjax):
            ax.set(xlim=(0, nsteps),
                   ylim=(0, 75))
            ax.set_ylabel(f'y_{row}', rotation=0, labelpad=20)
            t, = ax.plot([], [], label='True', c='c')
            p, = ax.plot([], [], label='Pred', c='m')
            ax.tick_params(labelbottom=False)
            ax.set_aspect(8)
        self.trjax[-1].set_xlabel('Time')
        self.trjax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                              fancybox=True, shadow=True, ncol=2)
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=15, metadata=dict(artist='Aaron Tuor'), bitrate=1800)
        self.ims = []

    def update_traj(self):
        with torch.no_grad():
            openloss, reg_error, X_out, Y_out, U_out = step(self.loop, self.data)
            Y_target = self.data[1]
            Yt = Y_target.squeeze().detach().cpu().numpy()
            Yp = Y_out.squeeze().detach().cpu().numpy()
            plots = []
            for k, ax in enumerate(self.trjax):
                plots.append(ax.plot(Yt[:, k], c='c', label=f'True')[0])
                plots.append(ax.plot(Yp[:, k], c='m', label=f'Pred')[0])
            return plots

    def __call__(self):
            mat = self.loop.model.fx.effective_W().detach().cpu().numpy()
            w, v = LA.eig(mat)
            self.ims.append([self.matax.imshow(mat),
                             self.eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))] +
                             self.update_traj())

    def make_and_save(self, filename):
        eig_ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, repeat_delay=3000)
        eig_ani.save(filename, writer=self.writer)


def setup():
    args = parse_args()
    os.system(f'mkdir {args.savedir}')
    if args.mlflow:
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run)
    params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
    if args.mlflow:
        mlflow.log_params(params)
    device = 'cpu'
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    return args, device


def data_setup(args, device):

    if args.system_data == 'datafile':
        Y, U, D, Ts = dataset.Load_data_sysID(args.datafile)  # load data from file
        plot.pltOL(Y, U=U, D=D)
    elif args.system_data == 'emulator':
        #  dataset creation from the emulator
        ninit = 0
        nsim = 1000
        building = emulators.Building_hf_ROM()  # instantiate building class
        building.parameters()  # load model parameters
        M_flow = emulators.Periodic(nx=building.n_mf, nsim=nsim, numPeriods=6, xmax=building.mf_max, xmin=building.mf_min,
                                    form='sin')
        DT = emulators.Periodic(nx=building.n_dT, nsim=nsim, numPeriods=9, xmax=building.dT_max, xmin=building.dT_min,
                                form='cos')
        D = building.D[ninit:nsim, :]
        U, X, Y = building.simulate(ninit, nsim, M_flow, DT, D)
        plot.pltOL(Y, U=U, D=D, X=X)

    if args.loop == 'open':
        # system ID or time series dataset
        Yp, Yf, Up, Uf, Dp, Df = dataset.make_dataset_ol(Y, U, D, nsteps=args.nsteps, device=device)
        train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Uf, Dp, Df]]
        dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Uf, Dp, Df]]
        test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Uf, Dp, Df]]

    elif args.loop == 'closed':
        # control loop dataset
        R = np.ones(Y.shape)
        Yp, Yf, Up, Dp, Df, Rf = dataset.make_dataset_cl(Y, U, D, R, nsteps=args.nsteps, device=device)
        train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Dp, Df, Rf]]
        dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Dp, Df, Rf]]
        test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Dp, Df, Rf]]

    nx, ny = Y.shape[1]*args.nx_hidden, Y.shape[1]
    print(ny)
    if U is not None:
        nu = U.shape[1]
    else:
        nu = 0
    if D is not None:
        nd = D.shape[1]
    else:
        nd = 0

    return train_data, dev_data, test_data, nx, ny, nu, nd, Y, [Yp, Yf, Up, Dp, Df]


def model_setup(args, device, nx, ny, nu, nd):
    linmap = linear.maps[args.linear_map]
    nonlinmap = {'linear': linmap,
                 'mlp': blocks.MLP,
                 'rnn': blocks.RNN,
                 'residual_mlp': blocks.ResMLP,
                 'sparse_residual_mlp': blocks.ResMLP}[args.nonlinear_map]

    fx = linmap(nx, nx, bias=args.bias).to(device)
    if args.ssm_type == 'BlockSSM':
        if args.nonlinear_map == 'sparse_residual_mlp':
            fy = blocks.ResMLP(nx, ny, bias=args.bias, hsizes=[nx] * 2,
                               Linear=linear.LassoLinear, skip=1).to(device)
        else:
            fy = nonlinmap(nx, ny, bias=args.bias, hsizes=[nx]*2, Linear=linmap, skip=1).to(device)

        if nu != 0:
            if args.nonlinear_map == 'sparse_residual_mlp':
                fu = blocks.ResMLP(nu, nx, bias=args.bias, hsizes=[nx] * 2,
                                   Linear=linear.LassoLinear, skip=1).to(device)
            else:
                fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx] * 2, Linear=linmap, skip=1).to(device)
        else:
            fu = None

        if nd != 0:
            fd = Linear(nd, nx).to(device)
        else:
            fd = None
        model = ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd).to(device)

    elif args.ssm_type == 'BlackSSM':
        fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx] * 3,
                            bias=args.bias, Linear=linmap, skip=1).to(device)
        fy = Linear(nx, ny, bias=args.bias).to(device)
        model = ssm.BlackSSM(nx, nu, nd, ny, fxud, fy).to(device)

    model.Q_dx, model.Q_dx_ud, model.Q_con_x, model.Q_con_u, model.Q_sub = \
        args.Q_dx, args.Q_dx_ud, args.Q_con_x, args.Q_con_u, args.Q_sub

    # TODO: dict
    if args.state_estimator == 'linear':
        estimator = estimators.LinearEstimator(ny, nx, bias=args.bias, linear=linmap)
    elif args.state_estimator == 'mlp':
        estimator = estimators.MLPEstimator(ny, nx, bias=args.bias, hsizes=[nx]*2,
                                            Linear=linmap, skip=1)
    elif args.state_estimator == 'rnn':
        estimator = estimators.RNNEstimator(ny, nx, bias=args.bias, num_layers=2,
                                            nonlinearity=F.gelu, Linear=linmap)
    elif args.state_estimator == 'kf':
        estimator = estimators.LinearKalmanFilter(model)
    else:
        estimator = estimators.FullyObservable()
    estimator = estimator.to(device)

    if args.loop == 'open':
        loop = loops.OpenLoop(model, estimator).to(device)
    elif args.loop == 'closed':
        policy = policies.LinearPolicy(nx, nu, nd, ny, args.nsteps).to(device)
        loop = loops.ClosedLoop(model, estimator, policy).to(device)
    #     TODO: shall we include constraints as an argument to loops
    #      or rather individually to model estimator and policy?
    #   loop = loops.ClosedLoop(model, estimator, policy, constraints).to(device)

    nweights = sum([i.numel() for i in list(loop.parameters()) if i.requires_grad])
    if args.mlflow:
        mlflow.log_param('Parameters', nweights)
    return model, loop


def main():
    args, device = setup()
    train_data, dev_data, test_data, nx, ny, nu, nd, Y, data = data_setup(args, device)
    # eigfig, eigwriter, eigax, matax, trjax, eigims = plot_setup(Y)
    model, loop = model_setup(args, device, nx, ny, nu, nd)
    anime = Animator(Y, data, loop)
    optimizer = torch.optim.AdamW(loop.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    elapsed_time = 0
    start_time = time.time()
    best_openloss = np.finfo(np.float32).max

    for i in range(args.epochs):
        model.train()
        loss, train_reg, _, _, _ = step(loop, train_data)

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            # MSE loss
            dev_loss, dev_reg, X_pred, Y_pred, U_pred = step(loop, dev_data)
            # open loop loss
            data_open = [d.transpose(0, 1).reshape(-1, 1, d.shape[-1]) if d is not None else d for d in dev_data]
            data_open = [d.transpose(0, 1) if d is not None else d for d in data_open]
            openloss, reg_error, X_out, Y_out, U_out = step(loop, data_open)

            if openloss < best_openloss:
                best_model = deepcopy(model.state_dict())
                best_openloss = openloss
            if args.mlflow:
                mlflow.log_metrics({'trainloss': loss.item(),
                                    'train_reg': train_reg.item(),
                                    'devloss': dev_loss.item(),
                                    'dev_reg': dev_reg.item(),
                                    'open': openloss.item(),
                                    'bestopen': best_openloss.item()}, step=i)
        if i % args.verbosity == 0:
            elapsed_time = time.time() - start_time
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}\topen: {openloss.item():10.8f}'
                  f'\tbestopen: {best_openloss.item():10.8f}\teltime: {elapsed_time:5.2f}s')

            if args.make_movie:
                anime()
                # with torch.no_grad():
                #     mat = fx.effective_W().detach().cpu().numpy()
                #     w, v = LA.eig(mat)
                #     eigims.append([matax.imshow(mat),
                #                    eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))] +
                #                    update_traj([Yp, Yf, Up, Uf, Dp, Df], loop, trjax))

        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()
    if args.make_movie:
        anime.make_and_save(os.path.join(args.savedir, f'{args.linear_map}2_transition_matrix.mp4'))

    plt.style.use('classic')
    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            loss, reg, X_out, Y_out, U_out = step(loop, dset)
            if args.mlflow:
                mlflow.log_metrics({f'nstep_{dname}_loss': loss.item(), f'nstep_{dname}_reg': reg.item()})
            Y_target = dset[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu))
            Ypred.append(Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(Y=np.concatenate(Ytrue),
                   Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred),
                   figname=os.path.join(args.savedir, 'nstep.png'))

        #  TODO: double check open loop evaluation
        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            data = [d.transpose(0, 1).reshape(1, -1, d.shape[-1]) if d is not None else d for d in dset]
            data = [d.transpose(0, 1) if d is not None else d for d in data]
            openloss, reg_error, X_out, Y_out, U_out = step(loop, data)
            print(f'{dname}_open_loss: {openloss}')
            if args.mlflow:
                mlflow.log_metrics({f'open_{dname}_loss': openloss.item(), f'open_{dname}_reg': reg_error.item()})
            Y_target = data[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu))
            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred), figname=os.path.join(args.savedir, 'open.png'))
        if args.make_movie:
            plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                  np.concatenate(Ypred).transpose(1, 0),
                                  figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                                  freq=args.freq)
        torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
        if args.mlflow:
            mlflow.log_artifacts(args.savedir)
            os.system(f'rm -rf {args.savedir}')


