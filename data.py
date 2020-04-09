import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import torch


def min_max_norm(M):
    """

    :param M:
    :return:
    """
    M_norm = (M - M.min(axis=0).reshape(1, -1)) / (M.max(axis=0) - M.min(axis=0)).reshape(1, -1)
    return np.nan_to_num(M_norm)


def control_profile_DAE(file_path='Reno_model_for_py.mat', dT_nominal_max=30,
                        dT_nominal_min=0, samples_day=288, sim_days=7):
    """

    :param file_path:
    :param dT_nominal_max:
    :param dT_nominal_min:
    :param samples_day:
    :param sim_days:
    :return:
    """
    file = loadmat(file_path)
    #    mass flow
    umax = file['umax']  # max heat per zone
    umin = file['umin']  # min heat per zone
    m_nominal_max = umax / 20  # maximal nominal mass flow l/h
    m_nominal_min = umin / 20  # minimal nominal mass flow
    m_flow_modulation_day = (
                0.5 + 0.5 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_day)))  # modulation of the pump
    m_flow_day = m_nominal_min + m_nominal_max * m_flow_modulation_day  # daily control profile
    M_flow = matlib.repmat(m_flow_day, 1, sim_days).T  # Sim_days control profile
    #    delta T
    dT_day = dT_nominal_min + (dT_nominal_max - dT_nominal_min) * (
                0.5 + 0.5 * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / samples_day)))  # daily control profile
    DT = matlib.repmat(dT_day, 1, sim_days).T  # Sim_days control profile
    return M_flow, DT


def disturbance(file_path='Reno_model_for_py.mat', n_sim=2016):
    """

    :param file_path:
    :param n_sim:
    :return:
    """
    return loadmat(file_path)['disturb'][:n_sim, :]  # n_sim X 3


class BuildingDAE:
    """

    """
    def __init__(self, file_path='Reno_model_for_py.mat', rom=False):
        """

        :param file_path: Location of matlab file with model parameters
        :param rom: (bool) Whether to run reduced order of full model
        """
        file = loadmat(file_path)
        self.Ts = file['Ts']  # sampling time
        self.TSup = file['TSup']  # supply temperature
        self.umax = file['umax']  # max heat per zone
        self.umin = file['umin']  # min heat per zone

        if rom:
            # reduced order linear model
            self.A = file['Ad_ROM']
            self.B = file['Bd_ROM']
            self.C = file['Cd_ROM']
            self.D = file['Dd_ROM']
            self.E = file['Ed_ROM']
            self.G = file['Gd_ROM']
            self.F = file['Fd_ROM']

        else:
            #  full order model
            self.A = file['Ad']
            self.B = file['Bd']
            self.C = file['Cd']
            self.D = file['Dd']
            self.E = file['Ed']
            self.G = file['Gd']
            self.F = file['Fd']

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.nd = self.E.shape[1]
        self.x = 0 * np.ones(self.nx, dtype=np.float32)  # initial conditions
        #         heat flow equation constants
        self.rho = 0.997  # density  of water kg/1l
        self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
        self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds

    def heat_flow(self, m_flow, dT):
        """

        :param m_flow:
        :param dT:
        :return:
        """
        U = m_flow * self.rho * self.cp * self.time_reg * dT
        return U

    def loop(self, nsim, M_flow, DT, D):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) Control profile matrix
        :param D: (ndarray, shape=(nsim, self.nd)) Disturbance matrix
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response matrices are aligned, i.e. X[k] is the state of the system that Y[k] is indicating
        """
        U = self.heat_flow(M_flow, DT)
        X = np.zeros((nsim + 1, self.nx))
        X[0] = self.x

        Y = np.zeros((nsim + 1, self.ny))  # output trajectory placeholders
        y = self.C * np.asmatrix(X[0, :]).T + self.F - 273.15
        Y[0, :] = y.flatten()

        for k in range(nsim):
            d = np.asmatrix(D[k, :]).T
            u = np.asmatrix(U[k, :]).T
            x = self.A * np.asmatrix(X[k, :]).T + self.B * u + self.E * d + self.G
            X[k + 1, :] = x.flatten()
            y = self.C * np.asmatrix(X[k + 1, :]).T + self.F - 273.15
            Y[k + 1, :] = y.flatten()

        # X = X + 20  # update states trajectories with initial condition of linearization
        return X, Y


def make_dataset(nsteps, device, norm='env', rom=False):
    """

    :param nsteps:
    :param device:
    :param norm:
    :param rom:
    :return:
    """
    M_flow, DT = control_profile_DAE(samples_day=288, sim_days=28)
    #    manual turnoffs
    M_flow[:, 3] = 0
    M_flow[:, 4] = 0
    M_flow[:, 5] = 0
    nsim = M_flow.shape[0]
    D = disturbance(n_sim=nsim)
    # TODO: select only subset of D

    building = BuildingDAE(rom=rom)
    X, Y = building.loop(nsim, M_flow, DT, D)

    if norm in ['env', 'all']:
        D, M_flow, DT = min_max_norm(D), min_max_norm(M_flow), min_max_norm(DT)
    if norm == 'all':
        X, Y = min_max_norm(X), min_max_norm(Y)
    target_Xresponse = X[1:][nsteps:]
    initial_states = X[:-1][nsteps:]
    target_Yresponse = Y[1:][nsteps:]
    initial_outputs = Y[:-1][:-nsteps]

    D_p, M_flow_p, DT_p = D[:-nsteps], M_flow[:-nsteps], DT[:-nsteps]
    D_f, M_flow_f, DT_f = D[nsteps:], M_flow[nsteps:], DT[nsteps:]

    # skip first week of data
    data = np.concatenate([initial_states, M_flow_f, DT_f, D_f, target_Xresponse,
                           target_Yresponse, initial_outputs, M_flow_p, DT_p, D_p], axis=1)[2016:]
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    data = torch.tensor(data, dtype=torch.float32).transpose(0, 1).to(device)  # nsteps X nsamples X nfeatures
    train_idx = (data.shape[1] // 3)
    dev_idx = train_idx * 2
    train_data = data[:, :train_idx, :]
    dev_data = data[:, train_idx:dev_idx, :]
    test_data = data[:, dev_idx:, :]

    return train_data, dev_data, test_data


