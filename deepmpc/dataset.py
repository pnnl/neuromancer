"""
Loading system ID datasets from mat files
"""
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import plot
import emulators


def min_max_norm(M):
    """
    :param M:
    :return:
    """
    M_norm = (M - M.min(axis=0).reshape(1, -1)) / (M.max(axis=0) - M.min(axis=0)).reshape(1, -1)
    return np.nan_to_num(M_norm)


#  TODO: save trained benchmark models from Matlab's System ID
#  TODO: write function to load csv files pre defined format as well
#  frame= load_data_sysID(file_path, type='csv') AT note: better to check file extension than add argument
#  make data generation based on pandas df as inputs
#  Data_sysID(identifier='pandas', data_file='frame', norm)
def load_data_from_file(system='aero'):
    """
    :param file_path: path to .mat file with dataset: y,u,d,Ts
    :return:
    """
    Y, U, D = None, None, None

    # list of available datasets
    systems_datapaths = {'tank': './datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat',
                         'vehicle3': './datasets/NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat',
                         'aero': './datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat',
                         'flexy_air': './datasets/Flexy_air/flexy_air_data.csv'}

    if system in systems_datapaths.keys():
        file_path = systems_datapaths[system]
    else:
        file_path = system

    file_type = file_path.split(".")[-1]
    if file_type == 'mat':
        file = loadmat(file_path)
        Y = file.get("y", None)  # outputs
        U = file.get("u", None)  # inputs
        D = file.get("d", None)  # disturbances
    elif file_type == 'csv':
        data = pd.read_csv(file_path)
        Y = data.filter(regex='y').values if data.filter(regex='y').values.size != 0 else None
        U = data.filter(regex='u').values if data.filter(regex='u').values.size != 0 else None
        D = data.filter(regex='d').values if data.filter(regex='d').values.size != 0 else None

    return Y, U, D

def load_data_from_emulator(system='LorenzSystem', nsim=None, ninit=None, ts=None):
    """
    dataset creation from the emulator
    """
    # list of available emulators
    systems = {'building_small': emulators.Building_hf_Small, # TODO: this one is not standardized
               # non-autonomous ODEs
               'CSTR': emulators.CSTR,
               'TwoTank': emulators.TwoTank,
               # autonomous chaotic ODEs
               'LorenzSystem': emulators.LorenzSystem,
               'Lorenz96': emulators.Lorenz96,
               'VanDerPol': emulators.VanDerPol,
               'ThomasAttractor': emulators.ThomasAttractor,
               'RosslerAttractor': emulators.RosslerAttractor,
               'LotkaVolterra': emulators.LotkaVolterra,
               'Brusselator1D': emulators.Brusselator1D,
               'ChuaCircuit': emulators.ChuaCircuit,
               'Duffing': emulators.Duffing,
               'UniversalOscillator': emulators.UniversalOscillator,
               # non-autonomous chaotic ODEs
               'HindmarshRose': emulators.HindmarshRose,
               # OpenAI gym environments
               'Pendulum-v0': emulators.GymWrapper,
               'CartPole-v1': emulators.GymWrapper,
               'Acrobot-v1': emulators.GymWrapper,
               'MountainCar-v0': emulators.GymWrapper,
               'MountainCarContinuous-v0': emulators.GymWrapper,
               # partially observable building state space models with external disturbances
               'Reno_full': emulators.BuildingEnvelope,
               'Reno_ROM40':  emulators.BuildingEnvelope,
               'RenoLight_full':  emulators.BuildingEnvelope,
               'RenoLight_ROM40':  emulators.BuildingEnvelope,
               'Old_full':  emulators.BuildingEnvelope,
               'Old_ROM40':  emulators.BuildingEnvelope,
               'HollandschHuys_full':  emulators.BuildingEnvelope,
               'HollandschHuys_ROM100':  emulators.BuildingEnvelope,
               'Infrax_full':  emulators.BuildingEnvelope,
               'Infrax_ROM100':  emulators.BuildingEnvelope
                }

    model = systems[system]()  # instantiate model class
    # simulate
    X, Y, U, D = None, None, None, None
    if isinstance(model, emulators.ODE_Autonomous):
        model.parameters()  # load model parameters
        X = model.simulate(nsim=nsim, ninit=ninit, ts=ts) # simulate open loop
        Y = X   # fully observable
        # plot.pltOL(Y=X)
        # plot.pltPhase(X=X)
    elif isinstance(model, emulators.ODE_NonAutonomous):
        model.parameters()  # load model parameters
        X, U = model.simulate(nsim=nsim, ninit=ninit, ts=ts) # simulate open loop
        Y = X   # fully observable
        # plot.pltOL(Y=X, U=U)
        # plot.pltPhase(X=X)
    elif isinstance(model, emulators.GymWrapper):
        model.parameters(system=system)  # load model parameters
        X, Reward, U = model.simulate(nsim=nsim, ninit=ninit, ts=ts)
        Y = Reward   # rewards used as outputs
        # plot.pltOL(Y=Y, X=X, U=U)
        # plot.pltPhase(X=X)
    elif isinstance(model, emulators.BuildingEnvelope):
        model.parameters(system=system)  # load model parameters
        X, Y, U, D, Q = model.simulate(nsim=nsim, ninit=ninit, ts=ts)
        # plot.pltOL(Y=Y, U=U, D=D, X=X)
        # plot.pltPhase(X=Y)
    # plt.savefig('./test/dataset.png')
    return Y, U, D

# TODO: old code, delete if load_data_from_emulator is stable
def load_data_from_emulator_building(system='building_small', nsim=1000, ninit=0):
    #  dataset creation from the emulator
    systems = {'building_small': emulators.Building_hf_Small,
               'building_ROM': emulators.BuildingEnvelope,
               'building_large': emulators.BuildingEnvelope}
    datafiles = {'building_small': './emulators/buildings/disturb.mat',
                 'building_ROM': './emulators/buildings/Reno_ROM40.mat',
                 'building_large': './emulators/buildings/Reno_full.mat'}
    building = systems[system]()  # instantiate building class
    building.parameters(file_path=datafiles[system])  # load model parameters
    M_flow = emulators.Periodic(nx=building.n_mf, nsim=nsim, numPeriods=6, xmax=building.mf_max, xmin=building.mf_min,
                                form='sin')
    DT = emulators.Periodic(nx=building.n_dT, nsim=nsim, numPeriods=9, xmax=building.dT_max, xmin=building.dT_min,
                            form='cos')
    D = building.D[ninit:nsim, :]
    U, X, Y = building.simulate(ninit, nsim, M_flow, DT, D)
    plot.pltOL(Y=Y, U=U, D=D, X=X)
    plt.savefig('./test/dataset.png')
    return Y, U, D


def batch_data(data, nsteps):
    """

    :param data: np.array shape=(total_time_steps, dim)
    :param nsteps: (int) n-step prediction horizon
    :return: torch.tensor shape=(nbatches, nsteps, dim)
    """
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    return torch.tensor(data, dtype=torch.float32).transpose(0, 1)  # nsteps X nsamples X nfeatures


def unbatch_data(data):
    return data.transpose(0, 1).reshape(-1, 1, data.shape[-1])


def split_train_test_dev(data):
    """
    exemplary use:
    # Yp_train, Yp_dev, Yp_test = split_train_test_dev(Yp)
    # Yf_train, Yf_dev, Yf_test = split_train_test_dev(Yf)
    """
    if data is not None:
        train_idx = (data.shape[1] // 3)
        dev_idx = train_idx * 2
        train_data = data[:, :train_idx, :]
        dev_data = data[:, train_idx:dev_idx, :]
        test_data = data[:, dev_idx:, :]
    else:
        train_data, dev_data, test_data = None, None, None
    return train_data, dev_data, test_data


def make_dataset_ol(Y, U, D, nsteps, device):
    """
    creates dataset for open loop system
    :param U: inputs
    :param Y: outputs
    :param nsteps: future windos (prediction horizon)
    :param device:
    :param norm:
    :return:
    """

    # Outputs: data for past and future moving horizons
    Yp, Yf = [batch_data(Y[:-nsteps], nsteps).to(device), batch_data(Y[nsteps:], nsteps).to(device)]
    # Inputs: data for past and future moving horizons
    Up = batch_data(U[:-nsteps], nsteps).to(device) if U is not None else None
    Uf = batch_data(U[nsteps:], nsteps).to(device) if U is not None else None
    # Disturbances: data for past and future moving horizons
    Dp = batch_data(D[:-nsteps], nsteps).to(device) if D is not None else None
    Df = batch_data(D[nsteps:], nsteps).to(device) if D is not None else None
    print(f'Yp shape: {Yp.shape} Yf shape: {Yf.shape}') if Y is not None else None
    print(f'Up shape: {Up.shape} Uf shape: {Uf.shape}') if U is not None else None
    print(f'Dp shape: {Dp.shape} Df shape: {Df.shape}') if D is not None else None
    return Yp, Yf, Up, Uf, Dp, Df


def make_dataset_cl(Y, U, D, R, nsteps, device):
    """
    creates dataset for closed loop system
    :param U: inputs
    :param Y: outputs
    :param nsteps: future windos (prediction horizon)
    :param device:
    :param norm:
    :return:
    """
    Yp, Yf, Up, Uf, Dp, Df = make_dataset_ol(Y, U, D, nsteps, device)
    Rf = batch_data(R[nsteps:], nsteps).to(device) if R is not None else None
    return Yp, Yf, Up, Dp, Df, Rf


def data_setup(args, device):

    if args.system_data == 'datafile':
        Y, U, D = load_data_from_file(system=args.system)  # load data from file
    elif args.system_data == 'emulator':
        Y, U, D = load_data_from_emulator(system=args.system, nsim=args.nsim)

    U = U.reshape(U.shape[0],-1) if U is not None else None
    Y = Y.reshape(Y.shape[0],-1) if Y is not None else None
    D = D.reshape(D.shape[0],-1) if D is not None else None
    U = min_max_norm(U) if ('U' in args.norm and U is not None) else None
    Y = min_max_norm(Y) if ('Y' in args.norm and Y is not None) else None
    D = min_max_norm(D) if ('D' in args.norm and D is not None) else None

    print(f'Y shape: {Y.shape}')
    plot.pltOL(Y=Y, U=U, D=D)
    plt.savefig('test.png')
    # system ID or time series dataset
    dataset = make_dataset_ol(Y, U, D, nsteps=args.nsteps, device=device)
    train_data = [split_train_test_dev(data)[0] for data in dataset]
    dev_data = [split_train_test_dev(data)[1] for data in dataset]
    test_data = [split_train_test_dev(data)[2] for data in dataset]

    nx, ny = Y.shape[1]*args.nx_hidden, Y.shape[1]
    nu = U.shape[1] if U is not None else 0
    nd = D.shape[1] if D is not None else 0

    return train_data, dev_data, test_data, nx, ny, nu, nd


if __name__ == '__main__':

    # Load data from files
    system_datasets = ['tank', 'vehicle3', 'aero', 'flexy_air']

    for system in system_datasets:
        Y, U, D = load_data_from_file(system)
        plot.pltOL(Y, U=U, D=D, figname='test.png')

        Yp, Yf, Up, Uf, Dp, Df = make_dataset_ol(Y, U, D, nsteps=32, device='cpu')
        plot.pltOL(np.concatenate([Yp[:, k, :] for k in range(Yp.shape[1])])[:1000],
                   Ytrain=np.concatenate([Yf[:, k, :] for k in range(Yf.shape[1])])[:1000], figname=f'{system}_align_test.png')

        R = np.ones(Y.shape)
        Yp, Yf, Up, Dp, Df, Rf = make_dataset_cl(Y, U, D, R, nsteps=5, device='cpu')

    # generate data from emulators
    systems = ['CSTR','TwoTank','LorenzSystem','Lorenz96','VanDerPol',
               'ThomasAttractor','RosslerAttractor','LotkaVolterra','Brusselator1D',
               'ChuaCircuit','Duffing','UniversalOscillator','HindmarshRose','Pendulum-v0',
               'CartPole-v1','Acrobot-v1','MountainCar-v0','MountainCarContinuous-v0',
               'Reno_full','Reno_ROM40','RenoLight_full','RenoLight_ROM40','Old_full',
               'Old_ROM40','HollandschHuys_full','HollandschHuys_ROM100','Infrax_full',
               'Infrax_ROM100']

    for system in systems:
        parser = argparse.ArgumentParser()
        data_group = parser.add_argument_group('DATA PARAMETERS')
        data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'], default='emulator',
                                help='source type of the dataset')
        data_group.add_argument('-system', default=system,
                                help='select particular dataset with keyword')
        data_group.add_argument('-nsteps', type=int, default=32,
                                help='Number of steps for open loop during training.')
        data_group.add_argument('-nsim', type=int, default=None,
                                help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                     'train, dev, and test will be split evenly from contiguous, sequential, '
                                     'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                     'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                     'None will invoke default values for each emulator')
        data_group.add_argument('-norm', type=str, default='UDY')
        data_group.add_argument('-nx_hidden', type=int, default=5, help='Number of hidden states per output')
        args = parser.parse_args()
        device = 'cpu'

        train_data, dev_data, test_data, nx, ny, nu, nd = data_setup(args, device)


