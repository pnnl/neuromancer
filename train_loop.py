"""
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
"""
"""
TODO: training options:
1, control via closed loop model
        trainable modules: SSM + estim + policy
        choices: fully/partially observable, w/wo measured disturbances d, SSM given or learned
2, sytem ID via open loop model
        trainable modules: SSM + estim
        choices: fully/partially observable, w/wo measured disturbances d
3, time series
        trainable modules: SSM + estim
"""
# python imports
import os
import argparse
from copy import deepcopy
import time
# ml imports
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# local imports
from plot import plot_trajectories
from linear import Linear
import dataset
import ssm
import estimators
import policies
import loops
import rnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cpu',
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-batchsize', type=int, default=-1)
    opt_group.add_argument('-epochs', type=int, default=500)
    opt_group.add_argument('-lr', type=float, default=0.003,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=16,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-datafile', default='./datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat',
                            help='Whether to use 40 variable reduced order model (opposed to 286 variable full model')
    data_group.add_argument('-norm', type=str, default='UDY')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-loop', type=str,
                             choices=['open', 'closed'], default='open')
    model_group.add_argument('-ssm_type', type=str, choices=['GT', 'BlockSSM', 'BlackSSM'], default='BlockSSM')
    model_group.add_argument('-nx_hidden', type=int, default=40, help='Number of hidden states')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['linear', 'mlp', 'rnn', 'kf'], default='linear')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    model_group.add_argument('-constr', action='store_true', default=True,
                             help='Whether to use constraints in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_u', type=float,  default=1e1, help='Relative penalty on hidden input constraints.')
    weight_group.add_argument('-Q_con_x', type=float,  default=1e1, help='Relative penalty on hidden state constraints.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=1e5, help='Relative penalty on maximal influence of u and d on hidden state in one time step.')
    weight_group.add_argument('-Q_dx', type=float,  default=1e2, help='Relative penalty on hidden state difference in one time step.')
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
    log_group.add_argument('-run', default='test',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-mlflow', default=False,
                           help='Using mlflow or not.')
    return parser.parse_args()

# single training step
def step(loop, data):
    if type(loop) is loops.OpenLoop:
        Yp, Yf, Up, Uf, Dp, Df = data
        X_pred, Y_pred, reg_error = loop(Yp, Up, Uf, Dp, Df)
        U_pred = None
    elif type(loop) is loops.ClosedLoop:
        Yp, Yf, Up, Dp, Df, Rf = data
        X_pred, Y_pred, U_pred, reg_error = loop(Yp, Up, Dp, Df, Rf)

    # print(Y_pred.shape, Yf.shape)
    # TODO: shall we create separate file losses.py with various types of loss functions?
    loss = Q_y * F.mse_loss(Y_pred.squeeze(), Yf.squeeze())

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

    ####################################
    ###### DATA SETUP ##################
    ####################################
    device = 'cpu'
    if args.gpu != 'cpu':
        device = f'cuda:{args.gpu}'

    Y, U, D, Ts = dataset.Load_data_sysID(args.datafile)  # load data from file

    if args.loop == 'open':
        # system ID dataset
        Yp, Yf, Up, Uf, Dp, Df = dataset.make_dataset_ol(Y, U, D, nsteps=args.nsteps, device=device)
        train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Uf, Dp, Df]]
        dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Uf, Dp, Df]]
        test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Uf, Dp, Df]]

    elif args.loop == 'closed':
        # control dataset
        R = np.ones(Y.shape)
        Yp, Yf, Up, Dp, Df, Rf = dataset.make_dataset_cl(Y, U, D, R, nsteps=args.nsteps, device=device)
        train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Dp, Df, Rf]]
        dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Dp, Df, Rf]]
        test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Dp, Df, Rf]]

    ####################################
    ###### DIMS SETUP ##################
    ####################################
    nx, ny = args.nx_hidden, Y.shape[1]
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

    #     TODO: fix the issue when fu and fd are not defined
    if args.ssm_type == 'BlockSSM':
        fx = Linear(nx, nx)
        fy = Linear(nx, ny)
        if nu != 0:
            fu = Linear(nu, nx)
        else:
            fu = None
        if nd != 0:
            fd = Linear(nd, nx)
        else:
            fd = None
        model = ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd).to(device)
    elif args.ssm_type == 'BlackSSM':
        fxud = Linear(nx+nu+nd, nx)
        fy = Linear(nx, ny)
        model = ssm.BlackSSM(nx, nu, nd, ny, fxud, fy).to(device)

    if args.state_estimator == 'linear':
        estimator = estimators.LinearEstimator(ny, nx, bias=args.bias)
    elif args.state_estimator == 'mlp':
        estimator = estimators.MLPEstimator(ny, nx, bias=args.bias, hsizes=[args.nx_hidden])
    elif args.state_estimator == 'rnn':
        estimator = estimators.RNNEstimator(ny, nx, bias=args.bias)
    elif args.state_estimator == 'kf':
        estimator = estimators.LinearKalmanFilter(model)
    else:
        estimator = estimators.FullyObservable()
    estimator.to(device)

    if args.loop == 'open':
        loop = loops.OpenLoop(model, estimator).to(device)
    elif args.loop == 'closed':
        policy = policies.LinearPolicy(nx, nu, nd, ny, args.nsteps).to(device)
        loop = loops.ClosedLoop(model, estimator, policy).to(device)

    nweights = sum([i.numel() for i in list(loop.parameters()) if i.requires_grad])
    if args.mlflow:
        mlflow.log_param('Parameters', nweights)

    ####################################
    ######OPTIMIZATION SETUP
    ####################################
    Q_y = args.Q_y/ny
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
        optimizer.zero_grad()
        losses = loss + train_reg
        losses.backward()
        optimizer.step()

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

    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        args.constr = False
        Q_y = 1.0
        #    TRAIN SET
        train_loss, train_reg, X_out, Y_out, U_out = step(loop, train_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_train_loss': train_loss.item(), 'nstep_train_reg': train_reg.item()})
        Y_target = train_data[1]
        ypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        ytrue = Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        #   DEV SET
        dev_loss, dev_reg, X_out, Y_out, U_out = step(loop, dev_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_dev_loss': dev_loss.item(),'nstep_dev_reg': dev_reg.item()})
        Y_target_dev = dev_data[1]
        devypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        devytrue = Y_target_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        #   TEST SET
        test_loss, test_reg, X_out, Y_out, U_out = step(loop, test_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_test_loss': test_loss.item(),'nstep_test_reg': test_reg.item()})
        Y_target_tst = test_data[1]
        testypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        testytrue = Y_target_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        plot_trajectories([np.concatenate([ytrue[:, k], devytrue[:, k], testytrue[:, k]])
                           for k in range(ypred.shape[1])],
                          [np.concatenate([ypred[:, k], devypred[:, k], testypred[:, k]])
                           for k in range(ypred.shape[1])],
                          ['$Y_1$', '$Y_2$', '$Y_3$', '$Y_4$', '$Y_5$', '$Y_6$'],
                          os.path.join(args.savedir, 'Y_nstep_large.png'))
