"""

"""
import argparse
from copy import deepcopy
import time

import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from plot import plot_trajectories
from data import Building_DAE, make_dataset
from ssm import SSM, PerronFrobeniusSSM, SVDSSM, SpectralSSM, SSMGroundTruth
from state_estimators import LinearEstimator, PerronFrobeniusEstimator, MLPEstimator, RNNEstimator, KalmanFilterEstimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-yonly', action='store_true',
                           help='Use only y prediction for loss update.')
    opt_group.add_argument('-batchsize', type=int, default=-1)
    opt_group.add_argument('-dropout', type=float, default=0.0)
    opt_group.add_argument('-epochs', type=int, default=1000)
    opt_group.add_argument('-lr', type=float, default=0.003,
                        help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=16,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-norm', action='store_true',
                            help='Whether to normalize U and D input')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-cell_type', type=str, choices=['true', 'vanilla', 'pf', 'svd', 'spectral'], default='vanilla')
    model_group.add_argument('-heatflow', type=str, choices=['black', 'grey', 'white'], default='white')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['true', 'linear', 'pf', 'mlp', 'rnn', 'rnn_spectral', 'rnn_svd', 'kf'], default='linear')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    model_group.add_argument('-n_hidden', type=int, default=16)
    model_group.add_argument('-nx_hidden', type=int, default=40, help='Number of hidden states')
    model_group.add_argument('-constr', action='store_true', default=True,
                             help='Whether to use constraints in the neural network models.')
    model_group.add_argument('-rnn_activation', type=str, choices=['gelu', 'relu', 'softplus'], default='gelu')
    model_group.add_argument('-rnn_init', type=str, choices=['basic', 'identity'], default='identity')
    model_group.add_argument('-rnn_bias', action='store_true')
    model_group.add_argument('-num_rnn_layers', type=int, default=1)
    model_group.add_argument('-num_layers', type=int, default=1)

    ##################
    # Weight PARAMETERS
    model_group = parser.add_argument_group('WEIGHT PARAMETERS')
    model_group.add_argument('-Q_con_u', type=float,  default=1e1, help='Relative penalty on hidden input constraints.')
    model_group.add_argument('-Q_con_x', type=float,  default=1e1, help='Relative penalty on hidden state constraints.')
    model_group.add_argument('-Q_dx_ud', type=float,  default=1e5, help='Relative penalty on maximal influence of u and d on hidden state in one time step.')
    model_group.add_argument('-Q_dx', type=float,  default=1e2, help='Relative penalty on hidden state difference in one time step.')
    model_group.add_argument('-Q_y', type=float,  default=1e0, help='Relative penalty on output tracking.')

    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='weights/garbage/',
                           help="Where should your trained model be saved")
    log_group.add_argument('-verbosity', type=int, default=100,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', default='test',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-mlflow', default = False,
                           help='Using mlflow or not.')
    
    return parser.parse_args()


def split_data(train_data, args):
    x0_in, M_flow_in, DT_in, D_in, x_response, Y_target, y0_in, M_flow_in_p, DT_in_p, D_in_p = (
    train_data[0, :, :nx],
    train_data[:, :, nx:nx + n_m],
    train_data[:, :, nx + n_m:nx + n_m + n_dT],
    train_data[:, :, nx + n_m + n_dT:nx + n_m + n_dT + nd],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny + n_m],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny + n_m:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny + n_m + n_dT],
    train_data[:, :,
    nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny + n_m + n_dT:nx + n_m + n_dT + nd + args.nx_hidden + args.nx_hidden + nu + nu + nx + ny + args.nx_hidden + nu + ny + n_m + n_dT + nd])
    return x0_in, M_flow_in, DT_in, D_in, x_response, Y_target, y0_in, M_flow_in_p, DT_in_p, D_in_p


def step(model, x0_in, y0_in, M_flow_in_p, DT_in_p, D_in_p, M_flow_in, DT_in, D_in):
    if args.state_estimator != 'true':
        x0_in = state_estimator(y0_in)
    else:
        x0_in = x0_in[0]
    X_pred, Y_pred, U_pred, regularization_error = model(x0_in, M_flow_in_p, DT_in_p, D_in_p,
                                                         M_flow_in, DT_in, D_in)
    if args.state_estimator != 'true':
        loss = Q_y * F.mse_loss(Y_pred.squeeze(), Y_target.squeeze())
    else:
        loss = F.mse_loss(X_pred.squeeze(), x_response.squeeze())
    if args.constr:
        loss += regularization_error
    return X_pred, Y_pred, U_pred, loss


if __name__ == '__main__':
    args = parse_args()

    ####################################
    ###### LOGGING SETUP
    ####################################
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
    ####################################
    ###### DATA SETUP
    ####################################
    models = {'true': SSMGroundTruth, 'vanilla': SSM, 'pf': PerronFrobeniusSSM,
              'svd': SVDSSM, 'spectral': SpectralSSM}
    building = Building_DAE()
    nx, nu, nd, ny, n_m, n_dT = building.nx, building.nu, building.nd, building.ny, building.nu, 1
    Q_y = args.Q_y/ny
    train_data, dev_data, test_data = make_dataset(args, device)
    
    x0_in, M_flow_in, DT_in, D_in, x_response, Y_target, y0_in, M_flow_in_p, DT_in_p, D_in_p = split_data(train_data, args)
    x0_dev, M_flow_dev,  DT_dev, D_dev, x_response_dev, Y_target_dev, y0_dev, M_flow_dev_p, DT_dev_p, D_dev_p = split_data(dev_data, args)
    x0_tst, M_flow_tst,  DT_tst,  D_tst, x_response_tst, Y_target_tst, y0_tst, M_flow_tst_p, DT_tst_p, D_tst_p = split_data(test_data, args)

    ####################################################
    ##### DYNAMICS MODEL AND STATE ESTIMATION SETUP ####
    ####################################################
    model = models[args.cell_type](nx, ny, n_m, n_dT, nu, nd, args.n_hidden, bias=args.bias, heatflow=args.heatflow,
                                   xmin=0, xmax=35, umin=-5000, umax=5000,
                                   Q_dx=args.Q_dx, Q_dx_ud=args.Q_dx_ud,
                                   Q_con_x=args.Q_con_x, Q_con_u=args.Q_con_u, Q_spectral=1e2)
    cells = {}
    if args.state_estimator == 'linear':
        state_estimator = LinearEstimator(nx, ny, bias=args.bias)
    elif args.state_estimator == 'pf':
        state_estimator = PerronFrobeniusEstimator()
    elif args.state_estimator == 'mlp':
        state_estimator = MLPEstimator()
    elif args.state_estimator in ['rnn', 'rnn_spectral', 'rnn_svd']:
        state_estimator = RNNEstimator(cell_type=cells[args.rnn_cell])
    elif args.state_estimator == 'kf':
        state_estimator = KalmanFilterEstimator()

    nweights = sum([i.numel() for i in list(model.parameters())])
    print(nweights, "parameters in the neural net.")
    if args.mlflow:    
        mlflow.log_param('Parameters', nweights)

    ####################################
    ######OPTIMIZATION SETUP
    ####################################
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    elapsed_time = 0
    start_time = time.time()
    best_dev = np.finfo(np.float32).max

    for i in range(args.epochs):
        model.train()
        X_pred, Y_pred, U_pred, loss = step(model, x0_in, y0_in, M_flow_in_p, DT_in_p, D_in_p, M_flow_in, DT_in, D_in)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)   # originally loss.backward() 
        optimizer.step()

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            X_pred, Y_pred, U_pred, dev_loss = step(model, x0_dev, y0_dev, M_flow_dev_p, DT_dev_p, D_dev_p,
                                                    M_flow_dev, DT_dev, D_dev)
            if dev_loss < best_dev:
                best_model = deepcopy(model.state_dict())
                best_dev = dev_loss
            if args.mlflow:    
                mlflow.log_metrics({'trainloss': loss.item(),
                                    'devloss': dev_loss.item(),
                                    'bestdev': best_dev.item()}, step=i)
        if i % args.verbosity == 0:
            elapsed_time = time.time() - start_time
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}\tdevloss: {dev_loss.item():10.8f}'
                  f'\tbestdev: {best_dev.item()}\teltime: {elapsed_time:5.2f}s')

    model.load_state_dict(best_model)
    ########################################
    ########## NSTEP TRAIN RESPONSE ########
    ########################################
    args.constr = False
    Q_y = 1.0
    #    TRAIN SET
    X_out, Y_out, U_out, train_loss = step(model, x0_in, y0_in, M_flow_in_p, DT_in_p,
                                           D_in_p, M_flow_in, DT_in, D_in)
    if args.mlflow:
        mlflow.log_metric('nstep_train_loss', train_loss.item())
    xpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, args.nx_hidden)
    xtrue = x_response.transpose(0, 1).detach().cpu().numpy().reshape(-1, nx)
    ypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    ytrue = Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    upred = U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu)

    #   DEV SET
    X_out, Y_out, U_out, dev_loss = step(model, x0_dev, y0_dev, M_flow_dev_p, DT_dev_p,
                                           D_dev_p, M_flow_dev, DT_dev, D_dev)
    if args.mlflow:
        mlflow.log_metric('nstep_dev_loss', dev_loss.item())
    devxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, args.nx_hidden)
    devxtrue = x_response_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, nx)
    devypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    devytrue = Y_target_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    devupred = U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu)

    #   TEST SET
    X_out, Y_out, U_out, test_loss = step(model, x0_tst, y0_tst, M_flow_tst_p, DT_tst_p,
                                           D_tst_p, M_flow_tst, DT_tst, D_tst)
    if args.mlflow:
        mlflow.log_metric('nstep_train_loss', train_loss.item())
    testxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, args.nx_hidden)
    testxtrue = x_response_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, nx)
    testypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    testytrue = Y_target_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
    testupred = U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu)

    plot_trajectories([np.concatenate([ytrue[:, k], devytrue[:, k], testytrue[:, k]])
                       for k in range(ypred.shape[1])],
                      [np.concatenate([ypred[:, k], devypred[:, k], testypred[:, k]])
                       for k in range(ypred.shape[1])],
                      ['$Y_1$', '$Y_2$', '$Y_3$', '$Y_4$', '$Y_5$', '$Y_6$'], 'Y_nstep_large.png')

    ########################################
    ########## OPEN LOOP RESPONSE ####
    ########################################
        
    def open_loop(model, data):
        data = torch.cat([data[:, k, :] for k in range(data.shape[1])]).unsqueeze(1)
        x0_in, M_flow_in, DT_in, D_in, x_response, Y_target, y0_in, M_flow_in_p, DT_in_p, D_in_p = split_data(data,args)
        X_pred, Y_pred, U_pred, regularization_error = model(y0_in, M_flow_in_p, DT_in_p, D_in_p, M_flow_in, DT_in, D_in)
        open_loss = F.mse_loss(Y_pred.squeeze(), Y_target.squeeze())

        return (open_loss.item(),
                X_pred.squeeze().detach().cpu().numpy(),
                Y_pred.squeeze().detach().cpu().numpy(),
                U_pred.squeeze().detach().cpu().numpy(),
                x_response.squeeze().detach().cpu().numpy(),
                Y_target.squeeze().detach().cpu().numpy(),
                M_flow_in.squeeze().detach().cpu().numpy(),
                DT_in.squeeze().detach().cpu().numpy(),
                D_in.squeeze().detach().cpu().numpy())
                

    openloss, xpred, ypred, upred, xtrue, ytrue, mflow_train, dT_train, d_train = open_loop(model, train_data)
    print(f'Train_open_loss: {openloss}')
    if args.mlflow:
        mlflow.log_metric('train_openloss', openloss)

    devopenloss, devxpred, devypred, devupred, devxtrue, devytrue, mflow_dev, dT_dev, d_dev = open_loop(model, dev_data)
    print(f'Dev_open_loss: {devopenloss}')
    if args.mlflow:
        mlflow.log_metric('dev_openloss', devopenloss)

    testopenloss, testxpred, testypred, testupred, testxtrue, testytrue, mflow_test, dT_test, d_test = open_loop(model, test_data)
    print(f'Test_open_loss: {testopenloss}')
    if args.mlflow:
        mlflow.log_metric('Test_openloss', testopenloss)
    
    plot_trajectories([np.concatenate([ytrue[:, k], devytrue[:, k], testytrue[:, k]])
    for k in range(ypred.shape[1])],
                   [np.concatenate([ypred[:, k], devypred[:, k], testypred[:, k]])
                    for k in range(ypred.shape[1])],  ['$Y_1$', '$Y_2$', '$Y_3$', '$Y_4$', '$Y_5$', '$Y_6$'],'./figs/y_open_test_large.png')
     
    fig, ax = plt.subplots(6, 1, figsize=(32, 32))
    ax[0].plot(np.concatenate([d_train, d_dev, d_test]))
    ax[1].plot(np.concatenate([mflow_train, mflow_dev, mflow_test]))
    ax[2].plot(np.concatenate([dT_train, dT_dev, dT_test]))
    ax[3].plot(np.concatenate([upred, devupred, testupred]))
    ax[4].plot(np.concatenate([xpred, devxpred, testxpred]))
    ax[5].plot(np.concatenate([xtrue, devxtrue, testxtrue]))
    plt.savefig('./figs/Raw_U_D.png')
    mlflow.log_artifact('./figs/Raw_U_D.png')