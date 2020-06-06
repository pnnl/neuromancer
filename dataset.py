"""
Loading system ID datasets from mat files
"""
from scipy.io import loadmat
import numpy as np
import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import plot
# TODO: add new imports to our environment
# import emulators

def min_max_norm(M):
    """
    :param M:
    :return:
    """
    M_norm = (M - M.min(axis=0).reshape(1, -1)) / (M.max(axis=0) - M.min(axis=0)).reshape(1, -1)
    return np.nan_to_num(M_norm)


# TODO: extend this function to load csv files pre defined format as well
#  frame= load_data_sysID(file_path, type='csv')
# make data generation based on pandas df as inputs
#  Data_sysID(identifier='pandas', data_file='frame', norm)
def Load_data_sysID(file_path='./datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat', norm='UDY'):
    """
    :param file_path: path to .mat file with dataset: y,u,d,Ts
    :return:
    """
    file = loadmat(file_path)
    Y = file.get("y", None)  # outputs
    U = file.get("u", None)  # inputs
    D = file.get("d", None)  # disturbances
    Ts = file.get("Ts", None)  # sampling time

    if 'U' in norm and U is not None:
        U = min_max_norm(U)
    if 'Y' in norm and Y is not None:
        Y = min_max_norm(Y)
    if 'D' in norm and D is not None:
        D = min_max_norm(D)
    print(Y.shape, U.shape)
    return Y, U, D, Ts


def data_batches(data, nsteps):
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    data = torch.tensor(data, dtype=torch.float32).transpose(0, 1)  # nsteps X nsamples X nfeatures
    return data

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
    # Yp, Yf = [data_batches(Y[:-1][:nsteps], nsteps).to(device), data_batches(Y[1:][nsteps:], nsteps).to(device)]

    Yp, Yf = [data_batches(Y[:-nsteps], nsteps).to(device), data_batches(Y[nsteps:], nsteps).to(device)]
    train_idx = (Yf.shape[1] // 3)
    dev_idx = train_idx * 2

    # Inputs: data for past and future moving horizons
    if U is not None:
        Up, Uf = [data_batches(U[:-nsteps], nsteps).to(device), data_batches(U[nsteps:], nsteps).to(device)]
    else:
        Up, Uf = None, None

    # Disturbances: data for past and future moving horizons
    if D is not None:
        Dp, Df = [data_batches(D[:-nsteps], nsteps).to(device), data_batches(D[nsteps:], nsteps).to(device)]
    else:
        Dp, Df = None, None
    print(Yp.shape, Yf.shape, Up.shape, Uf.shape)
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

    # Outputs: data for past and future moving horizons
    Yp, Yf = [data_batches(Y[:-nsteps], nsteps).to(device), data_batches(Y[nsteps:], nsteps).to(device)]
    train_idx = (Yf.shape[1] // 3)
    dev_idx = train_idx * 2

    # Inputs: data for past moving horizons
    if U is not None:
        Up, Uf = [data_batches(U[:-nsteps], nsteps).to(device), data_batches(U[nsteps:], nsteps).to(device)]
    else:
        Up, Uf = None, None

    # Disturbances: data for past and future moving horizons
    if D is not None:
        Dp, Df = [data_batches(D[:-nsteps], nsteps).to(device), data_batches(D[nsteps:], nsteps).to(device)]
    else:
        Dp, Df = None, None

    # References: data for future moving horizons
    if R is not None:
        Rp, Rf = [data_batches(R[:-nsteps], nsteps).to(device), data_batches(R[nsteps:], nsteps).to(device)]
    else:
        Rp, Rf = None, None

    return Yp, Yf, Up, Dp, Df, Rf


if __name__ == '__main__':
    datapaths = ['./datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat',
                 # './datasets/NLIN_SISO_predator_prey/PredPreyCrowdingData.mat',
                 # './datasets/NLIN_TS_pendulum/NLIN_TS_Pendulum.mat',
                 './datasets/NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat',
                 './datasets/NLIN_MIMO_CSTR/NLIN_MIMO_CSTR2.mat',
                 './datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat']


    for name, path in zip(['twotank', 'vehicle', 'reactor', 'aero'], datapaths):
        Y, U, D, Ts = Load_data_sysID(path)
        plot.pltOL(Y, U=U, D=D, figname='test.png')

        Yp, Yf, Up, Uf, Dp, Df = make_dataset_ol(Y, U, D, nsteps=32, device='cpu')
        plot.pltOL(np.concatenate([Yp[:, k, :] for k in range(Yp.shape[1])])[:1000],
                   Ytrain=np.concatenate([Yf[:, k, :] for k in range(Yf.shape[1])])[:1000], figname=f'{name}_align_test.png')

        R = np.ones(Y.shape)
        Yp, Yf, Up, Dp, Df, Rf = make_dataset_cl(Y, U, D, R, nsteps=5, device='cpu')


# #   TESTING dataset creation from the emulator
#     ninit = 0
#     nsim = 1000
#     building = emulators.Building_hf()   # instantiate building class
#     building.parameters()      # load model parameters
#     # generate input data
#     M_flow = emulators.Periodic(nx=building.n_mf, nsim=nsim, numPeriods=6, xmax=building.mf_max, xmin=building.mf_min, form='sin')
#     DT = emulators.Periodic(nx=building.n_dT, nsim=nsim, numPeriods=9, xmax=building.dT_max, xmin=building.dT_min, form='cos')
#     D = building.D[ninit:nsim,:]
#     # simulate open loop building
#     U, X, Y = building.simulate(ninit, nsim, M_flow, DT, D)
#     # plot trajectories
#     plot.pltOL(Y=Y, U=U, D=D, X=X)
#     # create datasets
#     Yp, Yf, Up, Uf, Dp, Df = make_dataset_ol(Y, U, D, nsteps=12, device='cpu')
#     R = 25*np.ones(Y.shape)
#     Yp, Yf, Up, Dp, Df, Rf = make_dataset_cl(Y, U, D, R, nsteps=12, device='cpu')
#     print(Yp.shape, Yf.shape, Up.shape, Dp.shape, Df.shape)

#   TODO: save trained benchmark models from Matlab's System ID
