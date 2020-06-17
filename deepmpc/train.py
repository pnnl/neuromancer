"""
Script for training block dynamics models for system identification.
Current block structure supported are black_box, hammerstein, and hammerstein weiner
Basic model options are:
    + prior on the linear maps of the neural network
    + state estimator
    + input non-linear map type for inputs and outputs
    + hidden state dimension
    + Whether to use affine or linear maps (bias term)
Basic data options are:
    + Load from a variety of premade data sequences
    + Load from a variety of emulators
    + Normalize input, output, or disturbance data
    + Nstep prediction horizon
Basic optimization options are:
    + Number of epochs to train on
    + Learn rate
Basic logging options are:
    + print to stdout
    + mlflow
    + weights and bias

More detailed description of options in the parse_args()
"""
# python imports
import os
import argparse
from copy import deepcopy
import time
# plotting imports
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ml imports
import mlflow
import torch
import numpy as np
# local imports
import plot
import dataset
import ssm
import estimators
import loops
import linear
import blocks


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
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'], default='emulator',
                            help='source type of the dataset')
    data_group.add_argument('-system', default='Reno_full',
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=None,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', type=str, default='UDY')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-ssm_type', type=str, choices=['hw', 'blackbox', 'hammerstein'], default='hw')
    model_group.add_argument('-nx_hidden', type=int, default=5, help='Number of hidden states per output')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()), default='linear')
    model_group.add_argument('-nonlinear_map', type=str, default='residual_mlp',
                             choices=['mlp', 'rnn', 'linear', 'residual_mlp'])
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_u', type=float,  default=0.0, help='Hidden input constraints penalty weight.')
    weight_group.add_argument('-Q_con_x', type=float,  default=0.0, help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=0.0,
                              help='Maximal influence of u and d on hidden state in one time step penalty weight.')
    weight_group.add_argument('-Q_dx', type=float,  default=0.0,
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_y', type=float,  default=1.0, help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float,  default=1.0, help='State estimator hidden prediction penalty weight')

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
    log_group.add_argument('-logger', choices=['mlflow', 'wandb', 'stdout'],
                           help='Logging setup to use')
    log_group.add_argument('-make_movie', action='store_true',
                           help='Make movie of best trace prediction at end of training')
    log_group.add_argument('-freq', type=int, default=10,
                           help='Frequency to create frames for reference tracking movie.')
    return parser.parse_args()


# single training step
def step(model, data):
    assert type(model) is loops.OpenLoop
    Yp, Yf, Up, Uf, Dp, Df = data
    X_pred, Y_pred, reg_error = model(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
    U_pred = Uf
    criterion = torch.nn.MSELoss()
    loss = criterion(Y_pred.squeeze(), Yf.squeeze())
    return loss, reg_error, X_pred, Y_pred, U_pred


def arg_setup():
    args = parse_args()
    os.system(f'mkdir {args.savedir}')
    if args.logger == 'wandb':
        import wandb
    if args.logger in ['mlflow', 'wandb']:
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run)
        params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
        mlflow.log_params(params)
    device = f'cuda:{args.gpu}' if (args.gpu is not None) else 'cpu'

    return args, device


def blackbox(args, linmap, nonlinmap, nx, nu, nd, ny):
    fxud = nonlinmap(nx + nu + nd, nx, hsizes=[nx] * 3,
                     bias=args.bias, Linear=linmap, skip=1)
    fy = linmap(nx, ny, bias=args.bias)
    return ssm.BlackSSM(nx, nu, nd, ny, fxud, fy)


def hammerstein(args, linmap, nonlinmap, nx, nu, nd, ny):
    fx = linmap(nx, nx, bias=args.bias)
    fy = linmap(nx, ny, bias=args.bias)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx] * 2, Linear=linmap, skip=1)
    fd = linmap(nd, nx)
    return ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)


def hw(args, linmap, nonlinmap, nx, nu, nd, ny):
    fx = linmap(nx, nx, bias=args.bias)
    fy = nonlinmap(nx, ny, bias=args.bias, hsizes=[nx]*2, Linear=linmap, skip=1)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx] * 2, Linear=linmap, skip=1) if nu != 0 else None
    fd = linmap(nd, nx) if nd != 0 else None
    return ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)


def model_setup(args, device, nx, ny, nu, nd):
    linmap = linear.maps[args.linear_map]
    nonlinmap = {'linear': linmap,
                 'mlp': blocks.MLP,
                 'rnn': blocks.RNN,
                 'residual_mlp': blocks.ResMLP}[args.nonlinear_map]

    # fx = linmap(nx, nx, bias=args.bias).to(device)
    ss_model = {'blackbox': blackbox,
                'hammerstein': hammerstein,
                'hw': hw}[args.ssm_type](args, linmap, nonlinmap, nx, nu, nd, ny)

    ss_model.Q_dx, ss_model.Q_dx_ud, ss_model.Q_con_x, ss_model.Q_con_u, ss_model.Q_sub = \
        args.Q_dx, args.Q_dx_ud, args.Q_con_x, args.Q_con_u, args.Q_sub

    estimator = {'linear': estimators.LinearEstimator,
                 'mlp': estimators.MLPEstimator,
                 'rnn': estimators.RNNEstimator,
                 'kf': estimators.LinearKalmanFilter}[args.state_estimator](ny, nx,
                                                                            bias=args.bias, hsized=[nx]*2,
                                                                            num_layers=2, Linear=linmap,
                                                                            ss_model=ss_model)
    model = loops.OpenLoop(model=ss_model, estim=estimator, Q_e=args.Q_e).to(device)

    nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
    if args.logger in ['mlflow', 'wandb']:
        mlflow.log_param('Parameters', nweights)
    return model


if __name__ == '__main__':
    args, device = arg_setup()
    train_data, dev_data, test_data, nx, ny, nu, nd = dataset.data_setup(args=args, device='cpu')
    model = model_setup(args, device, nx, ny, nu, nd)
    # Grab only first nsteps for previous observed states as input to state estimator
    data_open = [dev_data[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dev_data[1:]]
    anime = plot.Animator(data_open[1], model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    elapsed_time = 0
    start_time = time.time()
    best_openloss = np.finfo(np.float32).max
    best_model = deepcopy(model.state_dict())

    for i in range(args.epochs):
        model.train()
        loss, train_reg, _, _, _ = step(model, train_data)

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            # MSE loss
            dev_loss, dev_reg, X_pred, Y_pred, U_pred = step(model, dev_data)
            # open loop loss
            openloss, reg_error, X_out, Y_out, U_out = step(model, data_open)

            if openloss < best_openloss:
                best_model = deepcopy(model.state_dict())
                best_openloss = openloss
            if args.logger in ['mlflow', 'wandb']:
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
            if args.ssm_type != 'blackbox':
                anime(Y_out, data_open[1])

        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()
    anime.make_and_save(os.path.join(args.savedir, f'{args.linear_map}_transition_matrix.mp4'))

    plt.style.use('classic')
    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            loss, reg, X_out, Y_out, U_out = step(model, dset)
            if args.logger in ['mlflow', 'wandb']:
                mlflow.log_metrics({f'nstep_{dname}_loss': loss.item(), f'nstep_{dname}_reg': reg.item()})
            Y_target = dset[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
            Ypred.append(Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred) if U_out is not None else None,
                   figname=os.path.join(args.savedir, 'nstep.png'))

        Ytrue, Ypred, Upred = [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            data = [dset[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dset[1:]]
            # data = [dataset.unbatch_data(d) if d is not None else d for d in dset]
            openloss, reg_error, X_out, Y_out, U_out = step(model, data)
            print(f'{dname}_open_loss: {openloss}')
            if args.logger in ['mlflow', 'wandb']:
                mlflow.log_metrics({f'open_{dname}_loss': openloss.item(), f'open_{dname}_reg': reg_error.item()})
            Y_target = data[1]
            Upred.append(U_out.detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   U=np.concatenate(Upred) if U_out is not None else None,
                   figname=os.path.join(args.savedir, 'open.png'))
        if args.make_movie:
            plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                  np.concatenate(Ypred).transpose(1, 0),
                                  figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                                  freq=args.freq)
        torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
        if args.logger in ['mlflow', 'wandb']:
            mlflow.log_artifacts(args.savedir)
            os.system(f'rm -rf {args.savedir}')

