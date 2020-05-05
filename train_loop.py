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
# ml imports
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=20)
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
    model_group.add_argument('-ssm_type', type=str, choices=['GT', 'BlockSSM', 'BlackSSM'], default='BlackSSM')
    model_group.add_argument('-nx_hidden', type=int, default=10, help='Number of hidden states per output')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-linear_map', type=str,
                             choices=['pf', 'spectral', 'linear'], default='linear')
    model_group.add_argument('-nonlinear_map', type=str,
                             choices=['mlp', 'resnet', 'sparse_mlp', 'sparse_residual_mlp', 'linear'], default='mlp')
    model_group.add_argument('-nonlin', type=str,
                             choices=['relu', 'gelu'], default='gelu')
    model_group.add_argument('-bias', action='store_false', help='Whether to use bias in the neural network models.')


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

    # TODO: extent this to two control options: w/wo given model
    loss = F.mse_loss(Y_pred.squeeze(), Yf.squeeze())

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

    ####################################
    ###### DATA SETUP ##################
    ####################################
    device = 'cpu'
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'

    Y, U, D, Ts = dataset.Load_data_sysID(args.datafile)  # load data from file

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

    ####################################
    ###### DIMS SETUP ##################
    ####################################
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
              'pf': linear.PerronFrobeniusLinear}[args.linear_map]

    if args.ssm_type == 'BlockSSM':
        fx = linmap(nx, nx, bias=args.bias).to(device)
        if args.nonlinear_map == 'linear':
            fy = linmap(nx, ny, bias=args.bias).to(device)
        elif args.nonlinear_map == 'sparse_residual_mlp':
            fy = blocks.ResMLP(nx, ny, bias=args.bias, hsizes=[nx]*2,
                               Linear=linear.LassoLinear, skip=1).to(device)
        elif args.nonlinear_map == 'mlp':
            fy = blocks.MLP(nx, ny, bias=args.bias, hsizes=[nx]*2,
                               Linear=linmap).to(device)

        if nu != 0:
            if args.nonlinear_map == 'linear':
                fu = linmap(nu, nx, bias=args.bias).to(device)
            elif args.nonlinear_map == 'sparse_residual_mlp':
                fu = blocks.ResMLP(nu, nx, bias=args.bias, hsizes=[nx]*2,
                                   Linear=linear.LassoLinear, skip=1).to(device)
            elif args.nonlinear_map == 'mlp':
                fu = blocks.MLP(nu, nx, bias=args.bias, hsizes=[nx]*2,
                                   Linear=linear.LassoLinear).to(device)
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
        fxud = blocks.MLP(nx + nu + nd, nx, hsizes=[nx]*3,
                       bias=args.bias, nonlin=F.gelu).to(device)
        fy = Linear(nx, ny, bias=args.bias).to(device)
        model = ssm.BlackSSM(nx, nu, nd, ny, fxud, fy).to(device)

    if args.state_estimator == 'linear':
        estimator = estimators.LinearEstimator(ny, nx, bias=args.bias)
    elif args.state_estimator == 'mlp':
        estimator = estimators.MLPEstimator(ny, nx, bias=args.bias, hsizes=[nx]*2,
                                            Linear=linear.LassoLinear, skip=1)
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
        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        Ytrue, Ypred = [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            loss, reg, X_out, Y_out, U_out = step(loop, dset)
            if args.mlflow:
                mlflow.log_metric({f'nstep_{dname}_loss': loss.item(), 'nstep_{dname}_reg': reg.item()})
            Y_target = dset[1]
            Ypred.append(Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL_train(np.concatenate(Ytrue), np.concatenate(Ypred), figname=os.path.join(args.savedir, 'nstep.png'))

        Ytrue, Ypred = [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            data = [d.transpose(0, 1).view(1, -1, d.shape[-1]) if d is not None else d for d in dset]
            # print('true', [k.shape for k in data if k is not None])
            openloss, reg_error, X_out, Y_out, U_out = step(loop, data)
            print(f'{dname}_open_loss: {openloss}')
            if args.mlflow:
                mlflow.log_metrics({f'open_{dname}_loss': openloss, f'open_{dname}_reg': reg_error})
            Y_target = data[1]
            # print('true', Y_target.shape)
            # print('pred', Y_out.shape)
            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL_train(np.concatenate(Ytrue), np.concatenate(Ypred), figname=os.path.join(args.savedir, 'open.png'))
        torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
        if args.mlflow:
            mlflow.log_artifacts(args.savedir)
