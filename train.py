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
from data import make_dataset, Load_data_sysID
from ssm import BlockSSM
import estimators as se
# import policies as pol
from linear import Linear
import rnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cpu',
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-batchsize', type=int, default=-1)
    opt_group.add_argument('-epochs', type=int, default=5)
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
    # model_group.add_argument('-ssm_type', type=str, choices=['GT', 'linear', 'pf', 'svd', 'spectral'], default='linear')
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
    log_group.add_argument('-verbosity', type=int, default=1,
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
def step(model, state_estimator, data):
    Y_p, Y_f, U_p, U_f, D_p, D_f = data
    if args.state_estimator != 'GT':
        x0_in = state_estimator(Y_p, U_p, D_p)
    else:
        x0_in = Y_p[0]  # TODO: Is this what we want here?
    X_pred, Y_pred, regularization_error = model(x0_in, U_p, D_p)
    print(Y_pred.shape, Y_f.shape)
    loss = Q_y * F.mse_loss(Y_pred.squeeze(), Y_f.squeeze())
    regularization_error += args.Q_estim * state_estimator.reg_error()
    return X_pred, Y_pred, loss, regularization_error


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

    Y, U, D, Ts = Load_data_sysID(args.datafile)
    nx, nu, nd, ny = args.nx_hidden, U.shape[1], D.shape[1], Y.shape[1]
    train_data, dev_data, test_data = make_dataset(Y, U, D, Ts, args.nsteps, device)


    ####################################################
    ##### DYNAMICS MODEL AND STATE ESTIMATION SETUP ####
    ####################################################
    fx, fu, fd = [Linear(nk, nx) for nk in [nx, nu, nd]]
    fy = Linear(nx, ny)
    model = BlockSSM(nx, nu, nd, ny, fx, fu, fd, fy).to(device)
    if args.state_estimator == 'linear':
        state_estimator = se.LinearEstimator(ny, nx, bias=args.bias)
    elif args.state_estimator == 'mlp':
        state_estimator = se.MLPEstimator(ny, nx, bias=args.bias, hsizes=[args.nx_hidden])
    elif args.state_estimator == 'rnn':
        state_estimator = se.RNNEstimator(ny, nx, bias=args.bias)
    elif args.state_estimator == 'kf':
        state_estimator = se.KalmanFilterEstimator(model)
    else:
        state_estimator = se.LinearEstimator(ny, nx, bias=args.bias)
    state_estimator.to(device)

    nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
    if args.state_estimator != 'GT':
        nweights += sum([i.numel() for i in list(state_estimator.parameters()) if i.requires_grad])
    print(nweights, "parameters in the neural net.")
    if args.mlflow:
        mlflow.log_param('Parameters', nweights)

    ####################################
    ######OPTIMIZATION SETUP
    ####################################
    Q_y = args.Q_y/ny
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    elapsed_time = 0
    start_time = time.time()
    best_dev = np.finfo(np.float32).max

    for i in range(args.epochs):
        model.train()
        _, _, loss, train_reg = step(model, state_estimator, train_data)
        optimizer.zero_grad()
        losses = loss + train_reg
        losses.backward()
        optimizer.step()

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            X_pred, Y_pred, dev_loss, dev_reg = step(model, state_estimator, dev_data)
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
                  f'\tbestdev: {best_dev.item()}\teltime: {elapsed_time:5.2f}s')

    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        args.constr = False
        Q_y = 1.0
        #    TRAIN SET
        X_out, Y_out, train_loss, train_reg = step(model, state_estimator, train_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_train_loss': train_loss.item(), 'nstep_train_reg': train_reg.item()})
        Y_target = train_data[1]
        ypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        ytrue = Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        #   DEV SET
        X_out, Y_out, dev_loss, dev_reg = step(model, state_estimator, dev_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_dev_loss': dev_loss.item(),'nstep_dev_reg': dev_reg.item()})
        Y_target_dev = dev_data[1]
        devypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        devytrue = Y_target_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        #   TEST SET
        X_out, Y_out, test_loss, test_reg = step(model, state_estimator, test_data)
        if args.mlflow:
            mlflow.log_metric({'nstep_train_loss': train_loss.item(),'nstep_test_reg': test_reg.item()})
        Y_target_tst = test_data[1]
        testypred = Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)
        testytrue = Y_target_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)

        plot_trajectories([np.concatenate([ytrue[:, k], devytrue[:, k], testytrue[:, k]])
                           for k in range(ypred.shape[1])],
                          [np.concatenate([ypred[:, k], devypred[:, k], testypred[:, k]])
                           for k in range(ypred.shape[1])],
                          ['$Y_1$', '$Y_2$', '$Y_3$', '$Y_4$', '$Y_5$', '$Y_6$'],
                          os.path.join(args.savedir, 'Y_nstep_large.png'))
        # TODO: Open loop response
        # ########################################
        # ########## OPEN LOOP RESPONSE ##########
        # ########################################
        # def open_loop(model, data):
        #     data = torch.cat([data[:, k, :] for k in range(data.shape[1])]).unsqueeze(1)
        #     x0_in, M_flow_in, DT_in, D_in, x_response, Y_target, y0_in, M_flow_in_p, DT_in_p, D_in_p = split_data(data)
        #     if args.state_estimator == 'true':
        #         x0_in = x0_in[0]
        #     else:
        #         x0_in = state_estimator(y0_in, M_flow_in_p, DT_in_p, D_in_p)
        #     X_pred, Y_pred, U_pred, regularization_error = model(x0_in, M_flow_in, DT_in, D_in)
        #     open_loss = F.mse_loss(Y_pred.squeeze(), Y_target.squeeze())
        #     return (open_loss.item(),
        #             X_pred.squeeze().detach().cpu().numpy(),
        #             Y_pred.squeeze().detach().cpu().numpy(),
        #             U_pred.squeeze().detach().cpu().numpy(),
        #             x_response.squeeze().detach().cpu().numpy(),
        #             Y_target.squeeze().detach().cpu().numpy(),
        #             M_flow_in.squeeze().detach().cpu().numpy(),
        #             DT_in.squeeze().detach().cpu().numpy(),
        #             D_in.squeeze().detach().cpu().numpy())
        #
        #
        # openloss, xpred, ypred, upred, xtrue, ytrue, mflow_train, dT_train, d_train = open_loop(model, train_data)
        # print(f'Train_open_loss: {openloss}')
        # if args.mlflow:
        #     mlflow.log_metric('train_openloss', openloss)
        #
        # devopenloss, devxpred, devypred, devupred, devxtrue, devytrue, mflow_dev, dT_dev, d_dev = open_loop(model, dev_data)
        # print(f'Dev_open_loss: {devopenloss}')
        # if args.mlflow:
        #     mlflow.log_metric('dev_openloss', devopenloss)
        #
        # testopenloss, testxpred, testypred, testupred, testxtrue, testytrue, mflow_test, dT_test, d_test = open_loop(model, test_data)
        # print(f'Test_open_loss: {testopenloss}')
        # if args.mlflow:
        #     mlflow.log_metric('Test_openloss', testopenloss)
        #
        # plot_trajectories([np.concatenate([ytrue[:, k], devytrue[:, k], testytrue[:, k]])
        #                    for k in range(ypred.shape[1])],
        #                   [np.concatenate([ypred[:, k], devypred[:, k], testypred[:, k]])
        #                    for k in range(ypred.shape[1])], ['$Y_1$', '$Y_2$', '$Y_3$', '$Y_4$', '$Y_5$', '$Y_6$'],
        #                   os.path.join(args.savedir, 'y_open_test_large.png'))
        #
        # fig, ax = plt.subplots(6, 1, figsize=(32, 32))
        # ax[0].plot(np.concatenate([d_train, d_dev, d_test]))
        # ax[1].plot(np.concatenate([mflow_train, mflow_dev, mflow_test]))
        # ax[2].plot(np.concatenate([dT_train, dT_dev, dT_test]))
        # ax[3].plot(np.concatenate([upred, devupred, testupred]))
        # ax[4].plot(np.concatenate([xpred, devxpred, testxpred]))
        # ax[5].plot(np.concatenate([xtrue, devxtrue, testxtrue]))
        # plt.savefig(os.path.join(args.savedir, 'Raw_U_D.png'))
        # mlflow.log_artifacts(args.savedir)
        # os.system(f'rm -rf {args.savedir}')
