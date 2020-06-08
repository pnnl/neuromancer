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

def parse_args():
    parser = argparse.ArgumentParser()

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
    model_group.add_argument('-modelfile', type=str, help='Saved weights to load for analysis')
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
                             choices=['mlp', 'rnn', 'linear', 'residual_mlp', 'sparse_residual_mlp'], default='mlp')
    model_group.add_argument('-nonlin', type=str,
                             choices=['relu', 'gelu'], default='gelu')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-freq', type=int, default=1,
                           help='Frequency to create frames for reference tracking movie.')

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

    return loss, reg_error, X_pred, Y_pred, U_pred


if __name__ == '__main__':

    ####################################
    ###### DATA SETUP ##################
    ####################################
    args = parse_args()
    device = 'cpu'


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

    Yp, Yf, Up, Uf, Dp, Df = dataset.make_dataset_ol(Y, U, D, nsteps=args.nsteps, device=device)
    train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Uf, Dp, Df]]
    dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Uf, Dp, Df]]
    test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Uf, Dp, Df]]


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

    loop = loops.OpenLoop(model, estimator).to(device)



    plt.style.use('classic')
    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(torch.load(args.modelfile, map_location=torch.device('cpu')))
        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            loss, reg, X_out, Y_out, U_out = step(loop, dset)
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
            Y_target = data[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu))
            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred), figname=os.path.join(args.savedir, 'open.png'))
        plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                              np.concatenate(Ypred).transpose(1, 0),
                              figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                              freq=args.freq)

