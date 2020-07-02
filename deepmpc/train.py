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
import dynamics
import estimators
import policies
import loops
import linear
import blocks
import emulators

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
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'],
                            default='datafile',
                            help='source type of the dataset')
    data_group.add_argument('-system', default='flexy_air',
                            help='select particular dataset with keyword')
    # data_group.add_argument('-system', choices=list(emulators.systems.keys()), default='flexy_air',
    #                         help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=6000,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', choices=['UDY', 'U', 'Y', None], type=str, default='UDY')
    data_group.add_argument('-loop', type=str, choices=['closed', 'open'], default='open',
                            help='Defines open or closed loop for learning dynamics or control, respectively')
    data_group.add_argument('-adaptive', type=str, choices=[True, False], default=False,
                            help='extra option for closed loop'
                            'True = simultaneus policy optimization and system ID'
                            'False = policy optimization with given estimator and dynamics model parameters')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'hw', 'hammerstein', 'blocknlin'],
                             default='blocknlin')
    model_group.add_argument('-nx_hidden', type=int, default=5, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-policy', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='mlp')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='linear')
    model_group.add_argument('-nonlinear_map', type=str, default='mlp',
                             choices=['mlp', 'rnn', 'linear', 'residual_mlp'])
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_u', type=float,  default=0.2, help='Hidden input constraints penalty weight.')
    weight_group.add_argument('-Q_con_x', type=float,  default=0.2, help='Hidden state constraints penalty weight.')
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
    # assert type(model) is loops.OpenLoop
    criterion = torch.nn.MSELoss()
    if type(model) is loops.OpenLoop:
        Yp, Yf, Up, Uf, Dp, Df = data
        X_pred, Y_pred, reg_error = model(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
        U_pred = Uf
        loss = criterion(Y_pred.squeeze(), Yf.squeeze())
    elif type(model) is loops.ClosedLoop:
        Yp, Yf, Up, Dp, Df, Rf = data
        X_pred, Y_pred, U_pred, reg_error = model(Yp, Up, Dp, Df, Rf, nsamples=Yf.shape[0])
        loss = criterion(Y_pred.squeeze(), Rf.squeeze())
    return loss, reg_error, X_pred, Y_pred, U_pred, Df

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

# TODO: generalize hsized and num_layers
def model_setup(args, device, nx, ny, nu, nd):
    linmap = linear.maps[args.linear_map]
    nonlinmap = {'linear': linmap,
                 'mlp': blocks.MLP,
                 'rnn': blocks.RNN,
                 'residual_mlp': blocks.ResMLP}[args.nonlinear_map]
    # state space model setup
    dynamics_model = {'blackbox': dynamics.blackbox,
                'blocknlin': dynamics.blocknlin,
                'hammerstein': dynamics.hammerstein,
                'hw': dynamics.hw}[args.ssm_type](args, linmap, nonlinmap, nx, nu, nd, ny, args.n_layers)
    # state space model weights
    dynamics_model.Q_dx, dynamics_model.Q_dx_ud, dynamics_model.Q_con_x, dynamics_model.Q_con_u, dynamics_model.Q_sub = \
        args.Q_dx, args.Q_dx_ud, args.Q_con_x, args.Q_con_u, args.Q_sub
    # state estimator setup
    estimator = {'linear': estimators.LinearEstimator,
                 'mlp': estimators.MLPEstimator,
                 'rnn': estimators.RNNEstimator,
                 'kf': estimators.LinearKalmanFilter}[args.state_estimator](ny, nx,
                                                                            bias=args.bias,
                                                                            hsizes=[nx]*args.n_layers,
                                                                            num_layers=2, Linear=linmap,
                                                                            ss_model=dynamics_model)
    if args.loop == 'open':
        # open loop model setup
        model = loops.OpenLoop(model=dynamics_model, estim=estimator, Q_e=args.Q_e).to(device)
    elif args.loop == 'closed':
        # state estimator setup
        policy = {'linear': policies.LinearPolicy,
                     'mlp': policies.MLPPolicy,
                     'rnn': policies.RNNPolicy}[args.policy](nx, nu, nd, ny, N=args.nsteps,
                                                                                bias=args.bias,
                                                                                Linear=linmap,
                                                                                hsizes=[nx]*args.n_layers,
                                                                                num_layers=2
                                                                                )
        if not args.adaptive:
            # in case of non-adaptive policy optimization load trained system model and estimator
            dynamics_model.load_state_dict(torch.load(os.path.join(args.savedir, 'best_dynamics.pth')))
            estimator.load_state_dict(torch.load(os.path.join(args.savedir, 'best_estim.pth')))
            dynamics_model = dynamics_model.requires_grad_(False)
            estimator = estimator.requires_grad_(False)
        model = loops.ClosedLoop(model=dynamics_model, estim=estimator,
                                 policy=policy, Q_e=args.Q_e).to(device)
    nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
    if args.logger in ['mlflow', 'wandb']:
        mlflow.log_param('Parameters', nweights)
    return model


def model_perofmance(args, best_model, train_data, dev_data, test_data):
    plt.style.use('classic')
    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        Ytrue, Ypred, Upred, Rpred, Dpred = [], [], [], [], []
        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            loss, reg, X_out, Y_out, U_out, D_out = step(model, dset)
            if args.logger in ['mlflow', 'wandb']:
                mlflow.log_metrics({f'nstep_{dname}_loss': loss.item(), f'nstep_{dname}_reg': reg.item()})
            Y_target = dset[1]
            Upred.append(U_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
            Dpred.append(D_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, nd)) if D_out is not None else None
            Ypred.append(Y_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
            Ytrue.append(Y_target.transpose(0, 1).detach().cpu().numpy().reshape(-1, ny))
            Rpred.append(dset[-1].transpose(0, 1).detach().cpu().numpy().reshape(-1, ny)) if args.loop == 'closed' else None
        if args.loop == 'open':
            plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                       U=np.concatenate(Upred) if U_out is not None else None,
                       D=np.concatenate(Dpred) if D_out is not None else None,
                       figname=os.path.join(args.savedir, 'nstep_OL.png'))

        elif args.loop == 'closed':
            plot.pltCL(Y=np.concatenate(Ypred), R=np.concatenate(Rpred),
                       U=np.concatenate(Upred) if U_out is not None else None,
                       D=np.concatenate(Dpred) if D_out is not None else None,
                       figname=os.path.join(args.savedir, 'nstep_CL.png'))

        ########################################
        ########## OPEN LOOP RESPONSE ##########
        ########################################
        if args.loop == 'open':
            Ytrue, Ypred, Upred, Dpred = [], [], [], []
            for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
                data = [dset[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dset[1:]]
                # data = [dataset.unbatch_data(d) if d is not None else d for d in dset]
                # TODO: error here during closed loop does not handle well unbatched data
                openloss, reg_error, X_out, Y_out, U_out, D_out = step(model, data)
                print(f'{dname}_open_loss: {openloss}')
                if args.logger in ['mlflow', 'wandb']:
                    mlflow.log_metrics({f'open_{dname}_loss': openloss.item(), f'open_{dname}_reg': reg_error.item()})
                Y_target = data[1]
                Upred.append(U_out.detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
                Dpred.append(D_out.detach().cpu().numpy().reshape(-1, nd)) if D_out is not None else None
                # Upred.append(U_out.detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
                Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
                Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
            plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                       U=np.concatenate(Upred) if U_out is not None else None,
                       D=np.concatenate(Dpred) if D_out is not None else None,
                       figname=os.path.join(args.savedir, 'open.png'))
            if args.make_movie:
                plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                      np.concatenate(Ypred).transpose(1, 0),
                                      figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                                      freq=args.freq)

        # TODO: test and debug closed loop simulation in all variants
        ########################################
        ########## Closed LOOP RESPONSE ########
        ########################################
        # TODO: standalone closed loop simulation function
        if args.loop == 'closed':
            # initialize dynamics model in case of non-adaptive policy optimization
            if not args.adaptive:
                model.model.load_state_dict(torch.load(os.path.join(args.savedir, 'best_dynamics.pth')))
                model.estim.load_state_dict(torch.load(os.path.join(args.savedir, 'best_estim.pth')))

            # initialize disturbances and control actions
            if args.system_data == 'emulator':
                Y, U, D = dataset.load_data_from_emulator(system=args.system, nsim=args.nsim)
            elif args.system_data == 'datafile':
                Y, U, D = dataset.load_data_from_file(system=args.system, nsim=args.nsim)  # load data from file
            D, _, _ = dataset.min_max_norm(D) if D is not None else None, None, None
            Dpast = D[:-args.nsteps] if D is not None else None
            Dfuture = D[args.nsteps:] if D is not None else None
            Dp = torch.tensor(Dpast[0:args.nsteps, :]).reshape(args.nsteps, 1, nd) if D is not None else None
            Df = torch.tensor(Dfuture[0:args.nsteps, :]).reshape(args.nsteps, 1, nd) if D is not None else None
            Uinit = np.zeros([args.nsteps, nu])

            # initialize state trajectories
            if args.system_data == 'emulator':
                system_emualtor = emulators.systems[args.system]()
                if isinstance(system_emualtor, emulators.GymWrapper):
                    system_emualtor.parameters(system=args.system)
                elif isinstance(system_emualtor, emulators.BuildingEnvelope):
                    system_emualtor.parameters(system=args.system, linear=True)
                else:
                    system_emualtor.parameters()
                nsim = args.nsim if args.nsim is not None else system_emualtor.nsim
                # simulate system over nsteps to fill in Yp with Up = zeros
                X, Y, _, _ = system_emualtor.simulate(U=Uinit, nsim=args.nsteps, x0=system_emualtor.x0)  # simulate open loop
                x_denorm = X[-1, :]
                y_denorm = Y[-1, :]
                Yp = torch.tensor(Y).reshape(args.nsteps, 1, ny)
                Up = torch.tensor(Uinit).reshape(args.nsteps, 1, nu)
            elif args.system_data == 'datafile':
                nsim = args.nsim if args.nsim is not None else \
                    3 * train_data[0].shape[0] * train_data[0].shape[1]
                x0 = torch.zeros([1, nx])
                Up = torch.tensor(Uinit).reshape(args.nsteps, 1, nu).float()
                Xp, Yp, _ = model.model(x=x0, U=Up,
                                D=Df.float() if Df is not None else None, nsamples=1)
            # initialize references
            Ref = emulators.Periodic(nx=Y.shape[1], nsim=args.nsim,
                                   numPeriods=np.ceil(Y.shape[0] / 300).astype(int),
                                   xmax=0.4, xmin=0.7, form='sin')
            # Ref[:, 1:ny] = 0.5*Ref[:, 1:ny]
            Rf = torch.tensor(Ref[0:args.nsteps, :]).reshape(args.nsteps, 1, ny)
            # TODO: generalize for time varying reference with signals from emulators

            # simulate closed loop
            Y, Y_norm, Uopt, Uopt_norm = [], [], [], []
            for k in range(nsim-2*args.nsteps):
                Dp = torch.tensor(Dpast[k:args.nsteps+k, :]).reshape(args.nsteps, 1, nd) if D is not None else None
                Df = torch.tensor(Dfuture[k:args.nsteps+k, :]).reshape(args.nsteps, 1, nd) if D is not None else None
                x0, _ = model.estim(Yp.float(), Up.float(),
                                    Dp.float() if Dp is not None else None)
                U, _ = model.policy(x0, Df.float() if Df is not None else None, Rf.float())
                Up[0:-1,:,:] = Up[1:,:,:]
                Up[-1, :, :] = U[:, 0]
                uopt = U[:, 0].detach().numpy()
                Uopt_norm.append(uopt)
                # denormalize
                uopt = dataset.min_max_denorm(uopt, norms['Umin'], norms['Umax'])
                Uopt.append(uopt)

                # system dynamics
                if args.system_data == 'emulator':
                    x_denorm, y_denorm, _, _ = system_emualtor.simulate(U=uopt.reshape(-1, system_emualtor.nu),
                                                                        nsim=1, x0=x_denorm)  # simulate open loop
                    x_denorm = x_denorm.squeeze()
                    y_denorm = y_denorm.squeeze()
                    y_norm, _, _ = dataset.min_max_norm(y_denorm, norms['Ymin'], norms['Ymax'])
                elif args.system_data == 'datafile':
                    u = U[:, 0].reshape(1, nu)
                    x = Xp[-1, :, :] if k == 0 else xpred.reshape(1, nx)
                    xpred, ypred, _ = model.model(x=x, U=u,
                                      D=Df[:, 0].float() if Df is not None else None, nsamples=1)
                    y_norm = ypred.reshape(1, ny).detach().numpy()
                    y_denorm = dataset.min_max_denorm(y_norm, norms['Ymin'], norms['Ymax'])
                Y_norm.append(y_norm)
                Y.append(y_denorm)
                Yp[0:-1, :, :] = Yp[1:, :, :]
                Yp[-1, :, :] = torch.tensor(y_norm)
            # X_cl = np.asarray(X, dtype=np.float32)
            # U_cl = np.asarray(Uopt, dtype=np.float32)
            # Ref = dataset.min_max_denorm(Ref, norms['Ymin'], norms['Ymax'])
            # Ref_denorm = dataset.min_max_denorm(Ref, norms['Ymin'], norms['Ymax'])
            Y_cl = np.asarray(Y_norm, dtype=np.float32).reshape(-1, ny)
            U_cl = np.asarray(Uopt_norm, dtype=np.float32)
            plot.pltCL(Y=Y_cl, R=Ref, U=U_cl, figname=os.path.join(args.savedir, 'closed.png'))


if __name__ == '__main__':
    args, device = arg_setup()
    train_data, dev_data, test_data, nx, ny, nu, nd, norms = dataset.data_setup(args=args, device='cpu')
    model = model_setup(args, device, nx, ny, nu, nd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # TODO: Design user friendly high level interface
    #  for defining custom constrained optimal control problem
    # INSPIRATION: optimization toolboxes such as cvxpy, casadi and yalmip
    # https: // www.cvxpy.org /
    # https: // yalmip.github.io / example / standardmpc /
    # https: // web.casadi.org /

    # TODO: proposed format with custom problem definition:
    # data, dimensions = data_setup(args)
    # fx = ...      # Pytorch nn.Module
    # pi = ...      # Pytorch nn.Module
    # est = ...     # Pytorch nn.Module
    # con = ...     # Pytorch nn.Module
    # obj = ...     # torch.nn.functional
    # custom_models = {'dynamics': fx, 'policy': pi, 'estimator': est,
    #                   'constraints': con, 'objective': obj}
    # model = model_setup(args, dimensions, custom_models)
    # optimizer =  torch.optim.XY(model)
    # prob = problem(model, problem)
    # results = problem.train()

    # TODO: Future project - parametric programming for constrained optimization
    # IDEA: parametric solutions of constrained optimization problems formulated as
    # differentiable computational graph


    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    # TODO: hide the model loading and statistics inside some high level train init function
    elapsed_time = 0
    start_time = time.time()
    best_looploss = np.finfo(np.float32).max
    best_model = deepcopy(model.state_dict())
    best_estim = deepcopy(model.estim.state_dict())
    best_dynamics = deepcopy(model.model.state_dict())
    best_policy = deepcopy(model.policy.state_dict()) if args.loop == 'closed' else None

    # Grab only first nsteps for previous observed states as input to state estimator
    data_open = [dev_data[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dev_data[1:]]
    anime = plot.Animator(data_open[1], model)

    for i in range(args.epochs):
        model.train()
        loss, train_reg, _, _, _, _ = step(model, train_data)

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        # TODO: wrap this in  function called train_eval for high level API
        with torch.no_grad():
            model.eval()
            # MSE loss
            dev_loss, dev_reg, X_pred, Y_pred, U_pred, D_pred = step(model, dev_data)
            # open loop loss
            if args.loop == 'open':
                # TODO: error here during closed loop when using  data_open
                looploss, reg_error, X_out, Y_out, U_out, D_out = step(model, data_open)
            elif args.loop == 'closed':
                looploss = dev_loss
                reg_error = dev_reg
                # TODO: asses closed loop performance on the dev dataset

            if looploss < best_looploss:
                best_model = deepcopy(model.state_dict())
                best_estim = deepcopy(model.estim.state_dict())
                best_dynamics = deepcopy(model.model.state_dict())
                best_policy = deepcopy(model.policy.state_dict()) if args.loop == 'closed' else None
                best_looploss = looploss
            if args.logger in ['mlflow', 'wandb']:
                mlflow.log_metrics({'trainloss': loss.item(),
                                    'train_reg': train_reg.item(),
                                    'devloss': dev_loss.item(),
                                    'dev_reg': dev_reg.item(),
                                    'loop': looploss.item(),
                                    'best loop': best_looploss.item()}, step=i)
        if i % args.verbosity == 0:
            elapsed_time = time.time() - start_time
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}\topen: {looploss.item():10.8f}'
                  f'\tbestopen: {best_looploss.item():10.8f}\teltime: {elapsed_time:5.2f}s')
            if args.ssm_type not in ['blackbox', 'blocknlin']:
                anime(Y_out, data_open[1])

        # TODO: continue in high level API here
        optimizer.zero_grad()
        loss += train_reg.squeeze()
        loss.backward()
        optimizer.step()

    #     TODO: hide this animation function in high level API function called train_visualize or train_plot
    anime.make_and_save(os.path.join(args.savedir, f'{args.linear_map}_transition_matrix.mp4'))

    # TODO: hide this in function: train_log
    torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
    torch.save(best_estim, os.path.join(args.savedir, 'best_estim.pth'))
    torch.save(best_dynamics, os.path.join(args.savedir, 'best_dynamics.pth'))
    torch.save(best_policy, os.path.join(args.savedir, 'best_policy.pth')) if args.loop == 'closed' else None

    ########################################
    ########## SAVE ARTIFACTS ##############
    ########################################
    if args.logger in ['mlflow', 'wandb']:
        mlflow.log_artifacts(args.savedir)
        os.system(f'rm -rf {args.savedir}')

    ########################################
    ########## MODEL PEFROMANCE ############
    ########################################
    model_perofmance(args, best_model, train_data, dev_data, test_data)

