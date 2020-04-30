"""
Loading system ID datasets from mat files
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch


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
    # U, D = Y, Y  # TODO: remove this when we generalize
    Ts = file.get("Ts", None)  # sampling time

    # TODO: temporary fix
    if U is None:
        U = np.zeros([Y.shape[0], 1])
    if D is None:
        D = np.zeros([Y.shape[0], 1])

    if 'U' in norm and U is not None:
        U = min_max_norm(U)
    if 'Y' in norm and Y is not None:
        Y = min_max_norm(Y)
    if 'D' in norm and D is not None:
        D = min_max_norm(D)
    return Y, U, D, Ts


def Plot_data_sysID(Y, U, D, Ts):
    nrows = 3
    if U is None:
        nrows -= 1
    if D is None:
        nrows -= 1

    fig, ax = plt.subplots(nrows, 1, figsize=(20, 16))

    if nrows == 1:
        ax.plot(Y, linewidth=3)
        ax.grid(True)
        ax.set_title('Outputs', fontsize=24)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Y', fontsize=24)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
    else:
        ax[0].plot(Y, linewidth=3)
        ax[0].grid(True)
        ax[0].set_title('Outputs', fontsize=24)
        ax[0].set_xlabel('Time', fontsize=24)
        ax[0].set_ylabel('Y', fontsize=24)
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)

    if U is not None:
        ax[1].plot(U, linewidth=3)
        ax[1].grid(True)
        ax[1].set_title('Inputs', fontsize=24)
        ax[1].set_xlabel('Time', fontsize=24)
        ax[1].set_ylabel('U', fontsize=24)
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)

    if D is not None:
        idx = 2
        if U is None:
            idx -= 1
        ax[idx].plot(D, linewidth=3)
        ax[idx].grid(True)
        ax[idx].set_title('Disturbances', fontsize=24)
        ax[idx].set_xlabel('Time', fontsize=24)
        ax[idx].set_ylabel('D', fontsize=24)
        ax[idx].tick_params(axis='x', labelsize=22)
        ax[idx].tick_params(axis='y', labelsize=22)


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


def make_dataset(Y, U, D, Ts, nsteps, device):
    """
    :param U: inputs
    :param Y: outputs
    :param Ts: sampling time
    :param nsteps: prediction step
    :param device:
    :param norm:
    :return:
    """

    # Outputs: data for past and future moving horizons
    data = [Y[:-nsteps], Y[nsteps:]]
    sizes = [Y.shape[1], Y.shape[1]]
    # Inputs: data for past and future moving horizons
    if U is not None:
        data += [U[:-nsteps], U[nsteps:]]
        sizes += [U.shape[1], U.shape[1]]
    else: # TODO: remove
        data += [Y[:-nsteps], Y[nsteps:]]
        sizes += [Y.shape[1], Y.shape[1]]
    # Disturbances: data for past and future moving horizons
    if D is not None:
        data += [D[:-nsteps], D[nsteps:]]
        sizes += [D.shape[1], D.shape[1]]
    else: # TODO: remove
        data += [Y[:-nsteps], Y[nsteps:]]
        sizes += [Y.shape[1], Y.shape[1]]

    data = np.concatenate(data, axis=1)

    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    data = torch.tensor(data, dtype=torch.float32).transpose(0, 1).to(device)  # nsteps X nsamples X nfeatures
    train_idx = (data.shape[1] // 3)
    dev_idx = train_idx * 2
    train_data = data[:, :train_idx, :]
    dev_data = data[:, train_idx:dev_idx, :]
    test_data = data[:, dev_idx:, :]

    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)
    return ([train_data[:, :, start:end] for start, end in zip(starts, ends)],
            [dev_data[:, :, start:end] for start, end in zip(starts, ends)],
            [test_data[:, :, start:end] for start, end in zip(starts, ends)])


if __name__ == '__main__':
    datapaths = ['./datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat',
                 './datasets/NLIN_SISO_predator_prey/PredPreyCrowdingData.mat',
                 './datasets/NLIN_TS_pendulum/NLIN_TS_Pendulum.mat',
                 './datasets/NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat',
                 './datasets/NLIN_MIMO_CSTR/NLIN_MIMO_CSTR2.mat',
                 './datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat']

    for path in datapaths:
        Y, U, D, Ts = Load_data_sysID(path)

        Plot_data_sysID(Y, U, D, Ts)
        train_data, dev_data, test_data = make_dataset(Y, U, D, Ts, nsteps=6, device='cpu')
        Yp, Yf, Up, Uf, Dp, Df = make_dataset_ol(Y, U, D, nsteps=6, device='cpu')
        R = np.ones(Y.shape)
        Yp, Yf, Up, Dp, Df, Rf = make_dataset_cl(Y, U, D, R, nsteps=6, device='cpu')


# Q: do we want to separate U and D at this stage? for system ID it does not matter
#    we could separate them only in the control loop
#   TODO: save trained benchmark models from Matlab's System ID
