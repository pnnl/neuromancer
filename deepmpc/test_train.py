
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
import loops
import linear
import blocks
from train import arg_setup, model_setup, step

args, device = arg_setup()
args.epochs = 10
args.loop = 'open'

systems = {'tank': 'datafile', 'vehicle3': 'datafile', 'aero': 'datafile', 'flexy_air': 'datafile',
            'CSTR': 'emulator','TwoTank': 'emulator','LorenzSystem': 'emulator',
           'Lorenz96': 'emulator','VanDerPol': 'emulator', 'ThomasAttractor': 'emulator',
           'RosslerAttractor': 'emulator','LotkaVolterra': 'emulator','Brusselator1D': 'emulator',
               'ChuaCircuit': 'emulator','Duffing': 'emulator','UniversalOscillator': 'emulator',
           'HindmarshRose': 'emulator','Pendulum-v0': 'emulator',
               'CartPole-v1': 'emulator','Acrobot-v1': 'emulator','MountainCar-v0': 'emulator',
           'MountainCarContinuous-v0': 'emulator',
               'Reno_full': 'emulator','Reno_ROM40': 'emulator','RenoLight_full': 'emulator',
           'RenoLight_ROM40': 'emulator','Old_full': 'emulator',
               'Old_ROM40': 'emulator','HollandschHuys_full': 'emulator',
           'HollandschHuys_ROM100': 'emulator','Infrax_full': 'emulator',
               'Infrax_ROM100': 'emulator'}

# selected system for test - one per each class
# classes = {dataset, nonautonomous ODE, autonomous ODE, OpenAIgym, SSM}
test_systems = {'tank': 'datafile', 'CSTR': 'emulator', 'LorenzSystem': 'emulator',
                    'CartPole-v1': 'emulator', 'Reno_full': 'emulator'}

# TODO: issues with plot.Animator
for system, data_type in test_systems.items():
    for ssm_type in ['blackbox', 'hw', 'hammerstein', 'blocknlin']:
        for nonlinear_map in ['mlp', 'rnn', 'linear', 'residual_mlp']:
            for linear_map in ['pf', 'linear', 'softSVD']:
                for state_estimator in ['rnn', 'mlp', 'linear']:
                    print(ssm_type+nonlinear_map+linear_map+state_estimator)
                    args.nonlinear_map = nonlinear_map
                    args.linear_map = linear_map
                    args.ssm_type = ssm_type
                    args.state_estimator = state_estimator
                    args.system = system
                    args.system_data = data_type
                    args.nsteps = 2
                    args.nsim = None # alarm, does not work with arbitrary number
                    print(system)
                    model, best_model = None, None  # cleanup

                    # TODO: check args.nsim how it is handled
                    train_data, dev_data, test_data, nx, ny, nu, nd, norms = dataset.data_setup(args=args, device='cpu')
                    model = model_setup(args, device, nx, ny, nu, nd)
                    # Grab only first nsteps for previous observed states as input to state estimator
                    data_open = [dev_data[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dev_data[1:]]
                    # anime = plot.Animator(data_open[1], model)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

                    #######################################
                    ### N-STEP AHEAD TRAINING
                    #######################################
                    elapsed_time = 0
                    start_time = time.time()
                    best_looploss = np.finfo(np.float32).max
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
                            looploss, reg_error, X_out, Y_out, U_out = step(model, data_open)

                            if looploss < best_looploss:
                                best_model = deepcopy(model.state_dict())
                                best_looploss = looploss
                            if args.logger in ['mlflow', 'wandb']:
                                mlflow.log_metrics({'trainloss': loss.item(),
                                                    'train_reg': train_reg.item(),
                                                    'devloss': dev_loss.item(),
                                                    'dev_reg': dev_reg.item(),
                                                    'open': looploss.item(),
                                                    'bestopen': best_looploss.item()}, step=i)
                        if i % args.verbosity == 0:
                            elapsed_time = time.time() - start_time
                            print(f'epoch: {i:2}  loss: {loss.item():10.8f}\topen: {looploss.item():10.8f}'
                                  f'\tbestopen: {best_looploss.item():10.8f}\teltime: {elapsed_time:5.2f}s')
                            if args.ssm_type != 'blackbox':
                                pass
                                # anime(Y_out, data_open[1])

                        optimizer.zero_grad()
                        loss += train_reg.squeeze()
                        loss.backward()
                        optimizer.step()
                    # anime.make_and_save(os.path.join(args.savedir, f'{args.linear_map}_transition_matrix.mp4'))

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
                        # plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                        #            U=np.concatenate(Upred) if U_out is not None else None,
                        #            figname=os.path.join(args.savedir, 'nstep.png'))

                        Ytrue, Ypred, Upred = [], [], []
                        for dset, dname in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
                            data = [dset[0][:, 0:1, :]] + [dataset.unbatch_data(d) if d is not None else d for d in dset[1:]]
                            # data = [dataset.unbatch_data(d) if d is not None else d for d in dset]
                            looploss, reg_error, X_out, Y_out, U_out = step(model, data)
                            print(f'{dname}_open_loss: {looploss}')
                            if args.logger in ['mlflow', 'wandb']:
                                mlflow.log_metrics({f'open_{dname}_loss': looploss.item(), f'open_{dname}_reg': reg_error.item()})
                            Y_target = data[1]
                            Upred.append(U_out.detach().cpu().numpy().reshape(-1, nu)) if U_out is not None else None
                            Ypred.append(Y_out.detach().cpu().numpy().reshape(-1, ny))
                            Ytrue.append(Y_target.detach().cpu().numpy().reshape(-1, ny))
                        # plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                        #            U=np.concatenate(Upred) if U_out is not None else None,
                        #            figname=os.path.join(args.savedir, 'open.png'))
                        if args.make_movie:
                            plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                                  np.concatenate(Ypred).transpose(1, 0),
                                                  figname=os.path.join(args.savedir, f'open_movie_{args.linear_map}2.mp4'),
                                                  freq=args.freq)
                        torch.save(best_model, os.path.join(args.savedir, 'best_model.pth'))
                        if args.logger in ['mlflow', 'wandb']:
                            mlflow.log_artifacts(args.savedir)
                            os.system(f'rm -rf {args.savedir}')

