from scipy.io import loadmat
from gym import spaces, Env

import numpy as np
from neuromancer.psl.nonautonomous import systems, ODE_NonAutonomous

def disturbance(file='../../TimeSeries/disturb.mat', n_sim=8064):
    return loadmat(file)['D'][:, :n_sim].T # n_sim X 3


class GymWrapper(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator, U=None, ninit=None, nsim=None, ts=None, x0=None,
                 perturb=[lambda: 0. , lambda: 1.]):

        super().__init__()
        if isinstance(simulator, ODE_NonAutonomous):
            self.simulator = simulator
        else:
            self.simulator = systems[simulator](U=U, ninit=ninit, nsim=nsim, ts=ts, x0=x0, norm_func=norm_func)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=self.simulator.get_U().shape[-1], dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.simulator.x0.shape,dtype=np.float32)
        self.perturb = perturb

    def step(self, action):
        self.x = self.A*np.asmatrix(self.x).reshape(4, 1) + self.B*action.T + self.E*(self.D[self.tstep].reshape(3,1))
        self.y = (self.C * np.asmatrix(self.x)).flatten()
        self.tstep += 1
        observation = (self.y, self.x)[self.fully_observable].astype(np.float32)
        self.X_out = np.concatenate([self.X_out, np.array(self.x.reshape([1, 4]))])
        return np.array(observation).flatten(), self.reward(), self.tstep == self.X.shape[0], {'xout': self.X_out}

    def reset(self, dset='train'):

        self.tstep = 0
        observation = (self.y, self.x)[self.fully_observable].astype(np.float32)
        self.X_out = np.empty(shape=[0, 4])
        return np.array(observation).flatten()

    def render(self, mode='human'):
        print('render')

systems = {k: GymWrapper for k in GymWrapper.envs}

