import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import mlflow
import os
from scipy.io import loadmat
from gym import spaces, Env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, ACKTR
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
import numpy as np
import numpy.matlib
import tensorflow as tf
import time

# def min_max_norm(M):
#     return (M - M.min(axis=0).reshape(1, -1))/(M.max(axis=0) - M.min(axis=0)).reshape(1, -1)

def normalize(M, Mmin=None, Mmax=None):
        """
        :param M: (2-d np.array) Data to be normalized
        :param Mmin: (int) Optional minimum. If not provided is inferred from data.
        :param Mmax: (int) Optional maximum. If not provided is inferred from data.
        :return: (2-d np.array) Min-max normalized data
        """
        Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
        Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
        M_norm = (M - Mmin) / (Mmax - Mmin)
        return np.nan_to_num(M_norm)


def min_max_denorm(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = M*(Mmax - Mmin) + Mmin
    return np.nan_to_num(M_denorm)


def control_profile(max_input=4e3, samples_day=288, sim_days=7):
    """
    """
    U_day = max_input*np.sin(np.arange(0, 2*np.pi, 2*np.pi/samples_day)) #  samples_day
    U = np.tile(U_day, sim_days).reshape(-1, 1) # samples_day*sim_days
    return U


def disturbance(file='../../TimeSeries/disturb.mat', n_sim=8064):
    return loadmat(file)['D'][:, :n_sim].T  # n_sim X 3


class ToyBuilding(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, fully_observable=True, obs_norm=True, act_denorm=True,
                 w_mean=0.0, w_var=0.0, theta_mean=0.0, theta_var=0.0):
        super().__init__()
        self.nsim, nsim = 8064, 8064
        self.fully_observable = fully_observable
        self.act_denorm = act_denorm
        self.obs_norm = obs_norm
        self.w_mean = w_mean
        self.w_var = w_var
        self.theta_mean = theta_mean
        self.theta_var = theta_var
        self.nx, self.ny, self.nu, self.nd = 4, 1, 1, 3
        # self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.nu,), dtype=np.float32)
        self.action_space = spaces.Box(0, 5000, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                shape=[(self.ny+self.nd+self.ny+2*self.ny+2*self.nu,
                                        self.nx+self.nd+self.ny+2*self.ny+2*self.nu)[fully_observable]],
                                dtype=np.float32)
        self.A = np.matrix([[0.9950, 0.0017, 0.0000, 0.0031], [0.0007, 0.9957, 0.0003, 0.0031],
                            [0.0000, 0.0003, 0.9834, 0.0000], [0.2015, 0.4877, 0.0100, 0.2571]])
        self.B = np.matrix([[1.7586e-06], [1.7584e-06],
                            [1.8390e-10], [5.0563e-04]])
        self.E = np.matrix([[0.0002, 0.0000, 0.0000], [0.0002, 0.0000, 0.0000],
                            [0.0163, 0.0000, 0.0000], [0.0536, 0.0005, 0.0001]])
        self.C = np.matrix([[0.0, 0.0, 0.0, 1.0]])
        self.x = 20 * np.ones(4, dtype=np.float32)

        self.tstep = 0
        self.y = self.C*np.asmatrix(self.x).T
        self.U = control_profile(samples_day=288, sim_days=nsim//288)
        nsim = self.U.shape[0]
        self.D = disturbance(file='disturb.mat', n_sim=nsim)
        self.X, self.Y = self.loop(8064, self.U, self.D)
        self.x, self.y = self.X[2016], self.Y[2016]
        self.init_idx = {'train': 0, 'dev': 2015, 'test': 4031}
        self.X, self.Y = self.X[2016:], self.Y[2016:]
        self.X_out = np.empty(shape=[0, 4])
        print(self.X.shape)
        plot_trajectories([self.X[:, k] for k in range(self.X.shape[1])], [self.X[:, k] for k in range(self.X.shape[1])], ['$x_1$', '$x_2$', '$x_3$', '$x_4$'])

        #     constraints and references
        self.ymin_val = 19
        self.ymax_val = 25
        self.umin_val = 0
        self.umax_val = 5000
        self.s_ymin = self.ReLU(-self.y + self.ymin_val)
        self.s_ymax = self.ReLU(self.y - self.ymax_val)
        self.s_umin = self.ReLU(-np.array([0]) + self.umin_val)
        self.s_umax = self.ReLU(np.array([0]) - self.umax_val)

        samples_day = 288  # 288 samples per day with 5 min sampling
        # R_day_train = 15 + 10 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_day))  # daily control profile
        R_day_train = 20 + 2 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_day))  # daily control profile
        Sim_days = 35  # number of simulated days
        self.Ref = np.matlib.repmat(R_day_train, 1, Sim_days).T # Sim_days control profile
        self.reference = self.Ref[2016]
        # self.Ref_train = self.Ref[2016:4032].squeeze()   # ad hoc fix
        # self.Ref_train = 15 + 25 * np.random.rand(2016)

        # weights - the same as for deep MPC
        self.Q_con_u = 5e-7
        self.Q_con_x = 50
        self.Q_con_y = 50
        self.Q_u = 1e-7
        self.Q_u = 1e-6
        self.Q_ref = 20
        self.alpha_con = 0
    def xtrue(self, dset):
        start = self.init_idx[dset]
        return self.X[start:start+2016]

    def loop(self, nsim, U, D):
        """

        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) Control profile matrix
        :param D: (ndarray, shape=(nsim, self.nd)) Disturbance matrix
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response matrices are aligned, i.e. X[k] is the state of the system that Y[k] is indicating
        """
        Y = np.zeros((nsim+1, 1))  # output trajectory placeholders
        X = np.zeros((nsim+1, 4))
        X[0] = self.x

        for k in range(nsim):
            Y[k+1] = self.C*np.asmatrix(X[k]).T
            d = np.asmatrix(D[k]).T
            u = np.asmatrix(U[k]).T
            x = self.A*np.asmatrix(X[k]).T + self.B*u + self.E*d
            X[k+1] = x.flatten()
        return X, Y

    def obs_normalize(self, obs):
        ###### Normalize min max bounds
        ymin = 0
        ymax = 40
        umin = 0
        umax = 5000
        dmin = np.min(self.D, 0)
        dmax = np.max(self.D, 0)
        rmin = np.min(self.Ref, 0)
        rmax = np.max(self.Ref, 0)
        if self.fully_observable is True:
            ny = self.nx
        else:
            ny = self.ny
        y_norm = normalize(obs[0:ny], ymin, ymax)
        d_norm = normalize(obs[ny:ny+self.nd], dmin, dmax)
        r_norm = normalize(obs[ny + self.nd:ny + self.nd+self.ny], rmin, rmax)
        sy_norm = normalize(obs[ny + self.nd + self.ny:ny + self.nd + 3*self.ny], ymin, ymax)
        su_norm = normalize(obs[ny + self.nd + 3 * self.ny:], umin, umax)
        obs_norm = np.concatenate([y_norm, d_norm, r_norm, sy_norm, su_norm])
        return obs_norm

    def action_denorm(self, action):
        umin = 0
        umax = 5000
        action = min_max_denorm(action, umin, umax)
        return action

    def ReLU(self, x):
        return x * (x > 0)

    def step(self, action):
        if self.act_denorm is True:
            action = self.action_denorm(action)

        w = (self.w_mean - self.w_var) + (2 * self.w_var) * np.asmatrix(
            np.random.rand(self.nx, 1))  # additive uncertainty
        theta = (1 + self.theta_mean - self.theta_var) + (2 * self.theta_var) * np.asmatrix(
            np.random.rand(self.nx, self.nx))  # parametric uncertainty

        self.d = self.D[2016+self.tstep].reshape(3,1)
        # self.x = self.A*np.asmatrix(self.x).reshape(4, 1) + self.B*action.T + self.E*self.d
        self.x = np.multiply(theta, self.A)*np.asmatrix(self.x).reshape(4, 1) + self.B*action.T + self.E*self.d + w
        self.y = self.C * np.asmatrix(self.x)
        self.reference = self.Ref[2016+self.tstep]
        self.tstep += 1

        # Original features in deep MPC: xi = torch.cat((x, d, r, symin, symax, umin, umax), 1)
        y_obsv = np.concatenate([np.array(self.y).flatten(), self.d.flatten(), self.reference,
                                 np.array(self.s_ymin).flatten(), np.array(self.s_ymax).flatten(),
                                 np.array(self.s_umin).flatten(), np.array(self.s_umax).flatten()])
        x_obsv = np.concatenate([np.array(self.x).flatten(), self.d.flatten(), self.reference,
                                np.array(self.s_ymin).flatten(), np.array(self.s_ymax).flatten(),
                                np.array(self.s_umin).flatten(), np.array(self.s_umax).flatten()])
        # y_obsv = np.concatenate((np.array(self.y).flatten(), self.d.flatten(), self.reference))
        # x_obsv = np.concatenate((np.array(self.x).flatten(), self.d.flatten(), self.reference))
        observation = (y_obsv, x_obsv)[self.fully_observable].astype(np.float32)

        if self.obs_norm is True:
            observation = self.obs_normalize(observation)

        self.X_out = np.concatenate([self.X_out, np.array(self.x.reshape([1, 4]))])
        self.action = action
        self.s_ymin = self.ReLU(-self.y + self.ymin_val)
        self.s_ymax = self.ReLU(self.y - self.ymax_val)
        self.s_umin = self.ReLU(-action + self.umin_val)
        self.s_umax = self.ReLU(action - self.umax_val)

        return np.array(observation).flatten(), self.reward(), self.tstep == self.X.shape[0], {'xout': self.X_out}

    def reward(self):
        # return -np.mean((np.array(self.y - self.Y[self.tstep]))**2)
        con_penalties = self.Q_u * np.mean((np.array(self.action))**2) \
            + self.Q_con_y * np.mean((np.array(self.s_ymin))**2) \
            + self.Q_con_y * np.mean((np.array(self.s_ymax))**2)  \
            + self.Q_con_u * np.mean((np.array(self.s_umin))**2)  \
            + self.Q_con_u * np.mean((np.array(self.s_umax))**2)
        r = -self.Q_ref * np.mean((np.array(self.y - self.Ref[2016+self.tstep]))**2) \
            - self.alpha_con*con_penalties
        return r

    def reset(self, dset='train'):
        self.x = 15+5*np.random.randn(self.nx).reshape([-1,1])
        self.y = self.x[3].reshape([-1,1])
        self.reference = self.Ref[2016+self.init_idx[dset]]
        self.d = self.D[2016+self.init_idx[dset]]
        self.tstep = self.init_idx[dset]
        self.s_ymin = self.ReLU(self.y + self.ymin_val)
        self.s_ymax = self.ReLU(self.y - self.ymax_val)

        y_obsv = np.concatenate([np.array(self.y).flatten(), self.d.flatten(), self.reference,
                                np.array(self.s_ymin).flatten(), np.array(self.s_ymax).flatten(),
                                np.array([self.umin_val]), np.array([self.umax_val])])
        x_obsv = np.concatenate([np.array(self.x).flatten(), self.d.flatten(), self.reference,
                                np.array(self.s_ymin).flatten(), np.array(self.s_ymax).flatten(),
                                np.array([self.umin_val]), np.array([self.umax_val])])
        # y_obsv = np.concatenate((self.y.flatten(), self.reference))
        # x_obsv = np.concatenate((self.x.flatten(), self.reference))
        observation = (y_obsv, x_obsv)[self.fully_observable].astype(np.float32)
        self.X_out = np.empty(shape=[0, 4])
        return np.array(observation).flatten()

    def render(self, mode='human'):
        print('render')

def plot_control(R, Y, U, D, Ymax=None, Ymin=None, Umax=None, Umin=None, figname='test.png'):
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].plot(R, '--', label='R')
    ax[0].plot(Y, label='Y')
    ax[0].plot(Ymax, 'k--') if Ymax is not None else None
    ax[0].plot(Ymin, 'k--') if Ymin is not None else None
    ax[0].set(ylabel='Y')
    ax[1].plot(U, label='U')
    ax[1].plot(Umax, 'k--') if Umax is not None else None
    ax[1].plot(Umin, 'k--') if Umin is not None else None
    ax[1].set(ylabel='U')
    ax[2].plot(D, label='D')
    ax[2].set(ylabel='D')
    plt.tight_layout()
    plt.savefig(figname)


def plot_trajectories(traj1, traj2, labels, figname='test.png'):
    fig, ax = plt.subplots(len(traj1), 1)
    for row, (t1, t2, label) in enumerate(zip(traj1, traj2, labels)):
        if t2 is not None:
            ax[row].plot(t1.flatten(), label='True')
            ax[row].plot(t2.flatten(), '--', label='Pred')
        else:
            ax[row].plot(t1)
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    ylabel=label,
                    xlim=(0, len(t1)))
        ax[row].tick_params(labelbottom=False)
    ax[row].tick_params(labelbottom=True)
    ax[row].set_xlabel('Day')
    plt.tight_layout()
    plt.savefig(figname)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    # stable_baselines.ppo2.PPO2(policy, env, gamma=0.99, n_steps=128, ent_coef=0.01,
    #                            learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5,
    #                            lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
    #                            cliprange_vf=None, verbose=0, tensorboard_log=None,
    #                            _init_setup_model=True, policy_kwargs=None,
    #                            full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
    # https://arxiv.org/abs/1707.06347

    # stable_baselines.a2c.A2C(policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01,
    #                          max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99, momentum=0.0,
    #                          epsilon=1e-05, lr_schedule='constant', verbose=0, tensorboard_log=None,
    #                          _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
    #                          seed=None, n_cpu_tf_sess=None)
    # https://arxiv.org/abs/1602.01783

    # stable_baselines.acktr.ACKTR(policy, env, gamma=0.99, nprocs=None,
    #                              n_steps=20, ent_coef=0.01, vf_coef=0.25,
    #                              vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5,
    #                              kfac_clip=0.001, lr_schedule='linear', verbose=0,
    #                              tensorboard_log=None, _init_setup_model=True,
    #                              async_eigen_decomp=False, kfac_update=1, gae_lambda=None,
    #                              policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    # https://arxiv.org/abs/1708.05144
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=5)
    opt_group.add_argument('-lr', type=float, default=0.01,
                           help='Step size for gradient descent.')
    opt_group.add_argument('-alg', type=str, choices=['PPO2', 'A2C', 'ACKTR'], default='A2C')
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=128,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-constrained',  type=float, default=1.0,
                            help='Constrained yes or no.')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    # model_group.add_argument('-num_layers', type=int, default=1)
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    model_group.add_argument('-nx_hidden', type=int, default=10,
                            help='Number of hidden units.')

    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model be saved")
    log_group.add_argument('-modeldir', type=str, default='best_model',
                           help="Best saved models from previous runs")
    log_group.add_argument('-verbosity', type=int, default=10,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', default='test',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-logger', choices=['mlflow', 'wandb', 'stdout'],
                           help='Logging setup to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    ####################################
    ###### DATA SETUP
    ####################################
    env = ToyBuilding()
    # env = DummyVecEnv([ToyBuilding for i in range(4)])
    check_env(env)
    env.alpha_con = args.constrained

    n_hidden = args.nx_hidden
    algs = {'PPO2': PPO2, 'A2C': A2C, 'ACKTR': ACKTR}
    policies = {'PPO2': MlpPolicy, 'A2C': MlpPolicy, 'ACKTR': MlpPolicy}
    policy_kwargs = {'layers': [n_hidden, n_hidden]}
    model = algs[args.alg](policies[args.alg], env, n_steps=args.nsteps, verbose=0, learning_rate=args.lr, policy_kwargs=policy_kwargs)

    def openloop(dset='train', x0=None):
        if x0 is None:
            obs = env.reset(dset)
        else:
            obs = np.concatenate([x0, env.reference])
        rewards = []
        states = []
        disturbances = []
        references = []
        actions = []
        if env.fully_observable:
            ny = env.nx
        else:
            ny = env.ny
        for j in range(2016):
            action, _states = model.predict(obs)
            if action < 0:
                action = 0
            elif action > 1:
                action = 1
            env.obs_norm = True
            obs, reward, dones, info = env.step(action)
            env.tstep = env.tstep - 1
            # denormalize states and actions for plotting
            env.obs_norm = False
            obs_denorm, _, _, _ = env.step(action)
            rewards.append(reward)
            actions.append(env.action)
            states.append(obs_denorm[0:ny])
            disturbances.append(obs_denorm[ny:ny+env.nd])
            references.append(obs_denorm[ny+env.nd])
        mse_ref_open = -np.mean(np.array(rewards))
        return mse_ref_open, np.array(references), np.array(states), np.array(actions), np.array(disturbances)


    ##################################
    # SIMULATE
    ##################################
    Eval_runs = 20  # number of randomized closed-loop simulations, Paper value: 20
    param_uncertainty = True
    add_uncertainty = True
    show_plots = False
    if add_uncertainty:
        env.w_mean = 0
        env.w_var = 0.1
    else:
        env.w_mean = 0
        env.w_var = 0.0
    if param_uncertainty:
        env.theta_mean = 0
        env.theta_var = 0.01
    else:
        env.theta_mean = 0
        env.theta_var = 0.00

    # Load best model
    # best_model = model.load(os.path.join(args.modeldir, "RL_model_best_ACKTR.h5"))
    # https: // stable - baselines.readthedocs.io / en / master / modules / base.html
    best_model = model.load(os.path.join(args.modeldir, "RL_model_best_ACKTR_constr.h5"))
    # simulate best model
    model = best_model

    CPU_mean_time = np.zeros(Eval_runs)
    CPU_max_time = np.zeros(Eval_runs)

    MAE_constr_run = np.zeros(Eval_runs)
    MSE_ref_run = np.zeros(Eval_runs)
    MA_energy_run = np.zeros(Eval_runs)
    for run in range(0, Eval_runs):
        preds = []
        refs = []
        mses = []
        U = []
        D = []
        start_step_time = time.time()
        for dset in ['train', 'dev', 'test']:
            mse, ref, pred, actions, disturb = openloop(dset)
            preds.append(pred)
            refs.append(ref)
            mses.append(mse)
            U.append(actions)
            D.append(disturb)
        eval_time = time.time() - start_step_time
        Ymax = env.ymax_val*np.ones([2016,1])
        Ymin = env.ymin_val * np.ones([2016,1])
        Umax = env.umax_val * np.ones([2016,1])
        Umin = env.umin_val * np.ones([2016,1])

        # closed loop simulations plots
        if show_plots:
            plot_control(R=refs[0], Y=preds[0][:, 3], U=U[0], D=D[0],
                         Ymax=Ymax, Ymin=Ymin, Umax=Umax, Umin=Umin,
                         figname=os.path.join(args.savedir, 'control_train.png'))
            plot_control(R=refs[2], Y=preds[2][:, 3], U=U[2], D=D[2],
                         Ymax=Ymax, Ymin=Ymin, Umax=Umax, Umin=Umin,
                         figname=os.path.join(args.savedir, 'control_test.png'))

        MAE_constr_run[run] = np.mean(np.maximum((preds[2][:, 3] - Ymax.squeeze()), 0)) + \
                      np.mean(np.maximum((-preds[2][:, 3] + Ymin.squeeze()), 0))
        MSE_ref_run[run] = np.mean(np.square(preds[2][:, 3] - refs[0]))
        MA_energy_run[run] = np.mean(np.absolute(U[2]))

    MSE_ref = np.mean(MSE_ref_run)
    MA_energy = np.mean(MA_energy_run)
    MAE_constr = np.mean(MAE_constr_run)



