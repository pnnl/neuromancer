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
import linear
import blocks
import rnn
import emulators

from dataset import min_max_norm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=2)
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

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS') # TODO: These are not doing anything
    weight_group.add_argument('-Q_con_u', type=float,  default=1e1, help='Relative penalty on hidden input constraints.')
    weight_group.add_argument('-Q_con_x', type=float,  default=1e1, help='Relative penalty on hidden state constraints.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=1e1, help='Relative penalty on maximal influence of u and d on hidden state in one time step.')
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
    log_group.add_argument('-run', default='deepmpc',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-mlflow', action='store_true',
                           help='Using mlflow or not.')
    log_group.add_argument('-make_movie', action='store_true', help='Make movies with this flag.')
    log_group.add_argument('-freq', type=int, help='Frequency to create frames for reference tracking movie.', default=10)
    return parser.parse_args()


# single training step
def step(loop, data):
    Yp, Yf, Up, Uf, Dp, Df = data
    X_pred, Y_pred, reg_error = loop(Yp, Up, Uf, Dp, Df)
    U_pred = Uf
    criterion = torch.nn.MSELoss()
    loss = criterion(Y_pred.squeeze(), Yf.squeeze())
    return loss, reg_error, X_pred, Y_pred, U_pred


def plot_setup(Y):
    nsteps, ny = Y.shape
    plt.style.use('dark_background')
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(nrows=1+ny, ncols=2, figure=fig, width_ratios=[1,1],
                              height_ratios=[5] + [1]*ny)
    eigax = fig.add_subplot(gs[0, 1])
    eigax.set_title('State Transition Matrix Eigenvalues')
    eigax.set_ylim(-1.1, 1.1)
    eigax.set_xlim(-1.1, 1.1)
    eigax.set_aspect(1)

    matax = fig.add_subplot(gs[0, 0])
    matax.axis('off')
    matax.set_title('State Transition Matrix')

    trjax = [fig.add_subplot(gs[k, :]) for k in range(1, ny+1)]
    for row, ax in enumerate(trjax):
        ax.set(xlim=(0, nsteps),
               ylim=(0, 75))
        ax.set_ylabel(f'y_{row}', rotation=0, labelpad=20)
        t, = ax.plot([], [], label='True', c='c')
        p, = ax.plot([], [], label='Pred', c='m')
        ax.tick_params(labelbottom=False)
        ax.set_aspect(8)
    # trjax[-1].tick_params(labelbottom=True)
    trjax[-1].set_xlabel('Time')
    trjax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=2)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Aaron Tuor'), bitrate=1800)
    plt.savefig('test.png')
    return fig, writer, eigax, matax, trjax, []


def update_traj(data, loop, axes):
    data = [d.transpose(0, 1).reshape(1, -1, d.shape[-1]) if d is not None else d for d in data]
    openloss, reg_error, X_out, Y_out, U_out = step(loop, data)
    Y_target = data[1]
    Yt = Y_target.squeeze().detach().cpu().numpy()
    Yp = Y_out.squeeze().detach().cpu().numpy()
    plots = []

    for k, ax in enumerate(axes):
        plots.append(ax.plot(Yt[:, k], c='c', label=f'True')[0])
        plots.append(ax.plot(Yp[:, k], c='m', label=f'Pred')[0])
    return plots


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

    #  dataset creation from the emulator
    ninit = 0
    nsim = 8064
    building = emulators.Building_hf_ROM()  # instantiate building class
    building.parameters()  # load model parameters
    M_flow = emulators.Periodic(nx=building.n_mf, nsim=nsim, numPeriods=6, xmax=building.mf_max, xmin=building.mf_min,
                                form='sin')
    DT = emulators.Periodic(nx=building.n_dT, nsim=nsim, numPeriods=9, xmax=building.dT_max, xmin=building.dT_min,
                            form='cos')
    D = building.D[ninit:nsim, :]
    U, X, Y = building.simulate(ninit, nsim, M_flow, DT, D)
    if 'U' in args.norm:
        U = min_max_norm(U)
    if 'Y' in args.norm:
        Y = min_max_norm(Y)
    if 'D' in args.norm:
        D = min_max_norm(D)
    plot.pltOL(Y, U=U, D=D, X=X)

    Yp, Yf, Up, Uf, Dp, Df = dataset.make_dataset_ol(Y, U, D, nsteps=args.nsteps, device=device)
    train_data = [dataset.split_train_test_dev(data)[0] for data in [Yp, Yf, Up, Uf, Dp, Df]]
    dev_data = [dataset.split_train_test_dev(data)[1] for data in [Yp, Yf, Up, Uf, Dp, Df]]
    test_data = [dataset.split_train_test_dev(data)[2] for data in [Yp, Yf, Up, Uf, Dp, Df]]

    nx, ny, nu, nd = X.shape[1], Y.shape[1], U.shape[1], D.shape[1]
    eigfig, eigwriter, eigax, matax, trjax, eigims = plot_setup(Y)


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
    fy = linear.Linear(nx, ny, bias=args.bias).to(device)
    fu = nonlinmap(nu, nx, bias=args.bias, hsizes=[nx] * 2, Linear=linmap, skip=1).to(device)

    fd = Linear(nd, nx).to(device)
    model = ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd).to(device)
    # estimator = estimators.LinearKalmanFilter(model)
    estimator = estimators.RNNEstimator(ny, nx, bias=args.bias, num_layers=2,
                                        nonlinearity=F.gelu, Linear=linmap)
    estimator = estimator.to(device)
    loop = loops.OpenLoop(model, estimator).to(device)

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
                    mat = fx.effective_W().detach().cpu().numpy()
                    w, v = LA.eig(mat)
                    eigims.append([matax.imshow(mat),
                                   eigax.scatter(w.real, w.imag, alpha=0.5, c=plot.get_colors(len(w.real)))] +
                                   update_traj([Yp, Yf, Up, Uf, Dp, Df], loop, trjax))

        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()
    if args.make_movie:
        eig_ani = animation.ArtistAnimation(eigfig, eigims, interval=50, repeat_delay=3000)
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
        # torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
        # if args.mlflow:
        #     mlflow.log_artifacts(args.savedir)
        #     # os.system(f'rm -rf {args.savedir}')
