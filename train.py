"""
TODO: We are now best testing weights an bias. You should follow instructions here:
https://docs.wandb.com/library/integrations/mlflow

to update the conda environment to log to both mlflow and weights an biases.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
import linear
import blocks
import rnn
import emulators

import wandb
wandb.init(project="deepmpc")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=1000)
    opt_group.add_argument('-lr', type=float, default=0.003,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=16,
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
    model_group.add_argument('-nx_hidden', type=int, default=10, help='Number of hidden states per output')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-linear_map', type=str,
                             choices=['pf', 'spectral', 'linear', 'softSVD', 'sparse', 'split_linear'], default='linear')
    # TODO: spectral is quite expensive softSVD is much faster
    model_group.add_argument('-nonlinear_map', type=str,
                             choices=['mlp', 'resnet', 'rnn', 'linear', 'residual_mlp'], default='mlp')
    model_group.add_argument('-nonlin', type=str,
                             choices=['relu', 'gelu'], default='gelu')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS') # TODO: These are not doing anything
    weight_group.add_argument('-Q_con_u', type=float,  default=1e1, help='Relative penalty on hidden input constraints.')
    weight_group.add_argument('-Q_con_x', type=float,  default=1e1, help='Relative penalty on hidden state constraints.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=1e1, help='Relative penalty on maximal influence of u and d on hidden state in one time step.')
    weight_group.add_argument('-Q_dx', type=float,  default=1e1, help='Relative penalty on hidden state difference in one time step.')
    weight_group.add_argument('-Q_y', type=float,  default=1e0, help='Relative penalty on output tracking.')
    weight_group.add_argument('-Q_estim', type=float,  default=1e0, help='Relative penalty on state estimator regularization.')


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


if __name__ == '__main__':
    ####################################
    ###### LOGGING SETUP ###############
    ####################################
    args = parse_args()
    os.system(f'mkdir {args.savedir}')
    if args.mlflow:
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run)
    params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
    if args.mlflow:
        mlflow.log_params(params)

    ###############################
    ####### PLOTTING SETUP
    ###############################
    # Set up formatting for the movie files
    plt.style.use('dark_background')
    Writer = animation.writers['ffmpeg']
    eigwriter = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    eigfig, (eigax, matax) = plt.subplots(nrows=1, ncols=2)
    matax.axis('off')
    matax.set_title('State Transition Matrix $A$')
    eigax.set_title('$A$ Matrix Eigenvalues')
    eigfig.suptitle(f'{args.linear_map} Linear Parameter Evolution during Training')
    eigax.set_ylim(-1.1, 1.1)
    eigax.set_xlim(-1.1, 1.1)
    eigax.set_aspect(1)
    mat_ims = []
    eig_ims = []

    ####################################
    ###### DATA SETUP ##################
    ####################################
    device = 'cpu'
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'

    if args.system_data is 'datafile':
        Y, U, D, Ts = dataset.Load_data_sysID(args.datafile)  # load data from file
        plot.pltOL(Y, U=U, D=D)
    elif args.system_data is 'emulator':
        #  dataset creation from the emulator
        ninit = 0
        nsim = 1000
        building = emulators.Building_hf()  # instantiate building class
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
    if U is not None:
        nu = U.shape[1]
    else:
        nu = 0
    if D is not None:
        nd = D.shape[1]
    else:
        nd = 0

    ####################################################
    #####        OPEN / CLOSED LOOP MODEL           ####
    ####################################################
    linmap = {'linear': linear.Linear,
              'spectral': linear.SpectralLinear,
              'softSVD': linear.SVDLinear,
              'pf': linear.PerronFrobeniusLinear,
              'sparse': linear.LassoLinear,
              'split_linear': linear.StableSplitLinear}[args.linear_map]
    fx = linmap(nx, nx, bias=args.bias).to(device)
    if args.ssm_type == 'BlockSSM':
        if args.nonlinear_map == 'linear':
            fy = linmap(nx, ny, bias=args.bias).to(device)
        elif args.nonlinear_map == 'residual_mlp':
            fy = blocks.ResMLP(nx, ny, bias=args.bias, hsizes=[nx]*2,
                               Linear=linmap, skip=1).to(device)
        elif args.nonlinear_map == 'mlp':
            fy = blocks.MLP(nx, ny, bias=args.bias, hsizes=[nx]*2,
                            Linear=linmap).to(device)
        elif args.nonlinear_map == 'rnn':
            fy = blocks.RNN(nx, ny, bias=args.bias, hsizes=[nx]*2, Linear=linmap).to(device)
        if nu != 0:
            if args.nonlinear_map == 'linear':
                fu = linmap(nu, nx, bias=args.bias).to(device)
            elif args.nonlinear_map == 'residual_mlp':
                fu = blocks.ResMLP(nu, nx, bias=args.bias, hsizes=[nx] * 2,
                                   Linear=linmap, skip=1).to(device)
            elif args.nonlinear_map == 'mlp':
                fu = blocks.MLP(nu, nx, bias=args.bias, hsizes=[nx]*2,
                                   Linear=linear.LassoLinear).to(device)
            elif args.nonlinear_map == 'rnn':
                fu = blocks.RNN(nu, nx, bias=args.bias, hsizes=[nx] * 2, Linear=linmap).to(device)

        else:
            fu = None
        if nd != 0:
            fd = Linear(nd, nx).to(device)
        else:
            fd = None
        model = ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd).to(device)
    elif args.ssm_type == 'BlackSSM':
        # TODO: there is an error with RNN due to different output format than blocks
        # fxud = rnn.RNN(nx+nu+nd, nx, num_layers=3,
        #                bias=args.bias, nonlinearity=F.gelu)
        fxud = blocks.ResMLP(nx + nu + nd, nx, hsizes=[nx]*3,
                             bias=args.bias, nonlin=F.gelu).to(device)
        fy = Linear(nx, ny, bias=args.bias).to(device)
        model = ssm.BlackSSM(nx, nu, nd, ny, fxud, fy).to(device)

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

    ####################################
    ######OPTIMIZATION SETUP
    ####################################
    optimizer = torch.optim.AdamW(loop.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    elapsed_time = 0
    start_time = time.time()
    best_dev = np.finfo(np.float32).max

    for i in range(args.epochs):
        model.train()
        loss, train_reg, _, _, _ = step(loop, train_data)

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            dev_loss, dev_reg, X_pred, Y_pred, U_pred = step(loop, dev_data)
            if dev_loss < best_dev:
                best_model = deepcopy(model.state_dict())
                best_dev = dev_loss
            if args.mlflow:
                mlflow.log_metrics({'trainloss': loss.item(),
                                    'train_reg': train_reg.item(),
                                    'devloss': dev_loss.item(),
                                    'dev_reg': dev_reg.item(),
                                    'bestdev': best_dev.item()}, step=i)
        if i % args.verbosity == 0:
            elapsed_time = time.time() - start_time
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}\tdevloss: {dev_loss.item():10.8f}'
                  f'\tbestdev: {best_dev.item():10.8f}\teltime: {elapsed_time:5.2f}s')

            if args.make_movie:
                with torch.no_grad():
                    mat = fx.effective_W().detach().numpy()
                    w, v = LA.eig(mat)
                    eig_ims.append([matax.imshow(mat), eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))])
                    # mat_ims.append([matax.imshow(mat)])

        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()
    if args.make_movie:
        eig_ani = animation.ArtistAnimation(eigfig, eig_ims, interval=50, repeat_delay=3000)
        eig_ani.save(os.path.join(args.savedir, f'{args.linear_map}2_transition_matrix.mp4'), writer=eigwriter)

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
        plot.pltOL(np.concatenate(Ytrue),
                   Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred),
                   figname=os.path.join(args.savedir, 'nstep.png'))

        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            data = [d.transpose(0, 1).reshape(1, -1, d.shape[-1]) if d is not None else d for d in dset]
            openloss, reg_error, X_out, Y_out, U_out = step(loop, data)
            print(f'{dname}_open_loss: {openloss}')
            if args.mlflow:
                mlflow.log_metrics({f'open_{dname}_loss': openloss.item(), f'open_{dname}_reg': reg_error.item()})
            Y_target = data[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu))
            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred), figname=os.path.join(args.savedir, 'open.png'))
        if args.make_movie:
            plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                  np.concatenate(Ypred).transpose(1, 0),
                                  figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                                  freq=args.freq)
        torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
        if args.mlflow:
            mlflow.log_artifacts(args.savedir)
            # os.system(f'rm -rf {args.savedir}')
