import numpy as np
from scipy.io import loadmat
import torch


def min_max_norm(M):
    M_norm = (M - M.min(axis=0).reshape(1, -1)) / (M.max(axis=0) - M.min(axis=0)).reshape(1, -1)
    return np.nan_to_num(M_norm)


def control_profile_DAE(file_path='../Models/Matlab_matrices/Reno_model_for_py.mat', dT_nominal_max=30,
                        dT_nominal_min=0, samples_day=288, sim_days=7):
    """
    m_nominal_max: maximal nominal mass flow l/h
    m_nominal_min: minimal nominal mass flow
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
    M_flow = np.matlib.repmat(m_flow_day, 1, sim_days).T  # Sim_days control profile
    #    delta T
    dT_day = dT_nominal_min + (dT_nominal_max - dT_nominal_min) * (
                0.5 + 0.5 * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / samples_day)))  # daily control profile
    DT = np.matlib.repmat(dT_day, 1, sim_days).T  # Sim_days control profile
    return M_flow, DT


def disturbance(file_path='../Models/Matlab_matrices/Reno_model_for_py.mat', n_sim=2016):
    return loadmat(file_path)['disturb'][:n_sim, :]  # n_sim X 3


class Building_DAE:
    def __init__(self, file_path='../Models/Matlab_matrices/Reno_model_for_py.mat'):
        file = loadmat(file_path)

        #  full order model
        self.A = file['Ad']
        self.B = file['Bd']
        self.C = file['Cd']
        self.D = file['Dd']
        self.E = file['Ed']
        self.G = file['Gd']
        self.F = file['Fd']
        # reduced order linear model
        self.A_ROM = file['Ad_ROM']
        self.B_ROM = file['Bd_ROM']
        self.C_ROM = file['Cd_ROM']
        self.D_ROM = file['Dd_ROM']
        self.E_ROM = file['Ed_ROM']
        self.G_ROM = file['Gd_ROM']
        self.F_ROM = file['Fd_ROM']

        self.Ts = file['Ts']  # sampling time
        self.TSup = file['TSup']  # supply temperature
        self.umax = file['umax']  # max heat per zone
        self.umin = file['umin']  # min heat per zone

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.nd = self.E.shape[1]

        self.nx_ROM = self.A_ROM.shape[0]
        self.ny_ROM = self.C_ROM.shape[0]
        self.nu_ROM = self.B_ROM.shape[1]
        self.nd_ROM = self.E_ROM.shape[1]

        self.x = 0 * np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.x_ROM = 0 * np.ones(self.nx_ROM, dtype=np.float32)  # initial conditions

        #         heat flow equation constants
        self.rho = 0.997  # density  of water kg/1l
        self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
        self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds

    def heat_flow(self, m_flow, dT):
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

        X_ROM = np.zeros((nsim + 1, self.nx_ROM))
        X_ROM[0] = self.x_ROM

        Y_ROM = np.zeros((nsim + 1, self.ny_ROM))  # output trajectory placeholders
        y_ROM = self.C_ROM * np.asmatrix(X_ROM[0, :]).T + self.F_ROM - 273.15
        Y_ROM[0, :] = y_ROM.flatten()

        for k in range(nsim):
            d = np.asmatrix(D[k, :]).T
            u = np.asmatrix(U[k, :]).T

            x = self.A * np.asmatrix(X[k, :]).T + self.B * u + self.E * d + self.G
            X[k + 1, :] = x.flatten()

            x_ROM = self.A_ROM * np.asmatrix(X_ROM[k, :]).T + self.B_ROM * u + self.E_ROM * d + self.G_ROM
            X_ROM[k + 1, :] = x_ROM.flatten()

            y_ROM = self.C_ROM * np.asmatrix(X_ROM[k + 1, :]).T + self.F_ROM - 273.15
            Y_ROM[k + 1, :] = y_ROM.flatten()

            y = self.C * np.asmatrix(X[k + 1, :]).T + self.F - 273.15
            Y[k + 1, :] = y.flatten()

        X = X + 20  # update states trajectories with initial condition of linearization
        X_ROM = X_ROM + 20  # update states trajectories with initial condition of linearization
        return X, Y, X_ROM, Y_ROM


def make_dataset(args, device):
    M_flow, DT = control_profile_DAE(samples_day=288, sim_days=28)

    #    manual turnoffs
    M_flow[:, 3] = 0 * M_flow[:, 3]
    M_flow[:, 4] = 0 * M_flow[:, 4]
    M_flow[:, 5] = 0 * M_flow[:, 5]

    nsim = M_flow.shape[0]
    D = disturbance(n_sim=nsim)
    # TODO: select only subset of D

    building = Building_DAE()

    D_scale, M_flow_scale, DT_scale = min_max_norm(D), min_max_norm(M_flow), min_max_norm(DT)
    # TODO: we may want to try unscaled version especially for the white box SSM
    X, Y, X_ROM, Y_ROM = building.loop(nsim, M_flow, DT, D)

    target_Xresponse = X[1:][args.nsteps:]
    initial_states = X[:-1][args.nsteps:]
    target_Yresponse = Y[1:][args.nsteps:]
    initial_outputs = Y[:-1][:-args.nsteps]

    D_scale_p, M_flow_scale_p, DT_scale_p = D_scale[:-args.nsteps], M_flow_scale[:-args.nsteps], DT_scale[:-args.nsteps]
    D_scale_f, M_flow_scale_f, DT_scale_f = D_scale[args.nsteps:], M_flow_scale[args.nsteps:], DT_scale[args.nsteps:]

    data = np.concatenate(
        [initial_states, M_flow_scale_f, DT_scale_f, D_scale_f, target_Xresponse,
         target_Yresponse, initial_outputs, M_flow_scale_p, DT_scale_p, D_scale_p], axis=1)[2016:]
    nsplits = (data.shape[0]) // args.nsteps
    leftover = (data.shape[0]) % args.nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    data = torch.tensor(data, dtype=torch.float32).transpose(0, 1).to(device)  # nsteps X nsamples X nfeatures
    train_idx = (data.shape[1] // 3)
    dev_idx = train_idx * 2
    train_data = data[:, :train_idx, :]
    dev_data = data[:, train_idx:dev_idx, :]
    test_data = data[:, dev_idx:, :]

    return train_data, dev_data, test_data


