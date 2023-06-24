'''

'''
import numpy as np
import random as rd
import gymnasium

class GymPsl:
    """
    Wraps a gymnasium environment in a psl environment
    """
    def __init__(self, gym_env: gymnasium.Env, nsim=1001, ninit=0, ts=0.1, seed=59):
        self.gym_env = gym_env
        self.set_seed(seed)
        self.x0 = self.gym_env.observation_space.sample()
        self.nx = self.x0.shape[0]
        self.nu = self.gym_env.action_space.shape
        if len(self.nu)==0:
            self.nu = 1
        else:
            self.nu = np.prod(self.nu)
        self.nsim, self.ninit, self.ts = nsim, ninit, ts
        self.ts = ts

    def set_seed(self, seed, set_global=False):
        if set_global:
            np.random.seed(seed)
            rd.seed(seed)
            # torch.manual_seed(seed)

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.gym_env.np_random = self.rng

    def simulate(self, U=None, ninit=None, nsim=None, Time=None, ts=None, x0=None, show_progress=False, seed=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :return: dict with keys: X, Y, U
        """
        if seed is not None:
            self.set_seed(seed)

        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if ts is None:
            ts = self.ts
        if U is None:
            U = self.get_U(nsim)
        if Time is None:
            Time = np.arange(0, nsim+1) * ts + ninit

        X = [self.get_x0()] # get_x0 also calls gym_env.reset() which is necessary to initialize gym_env
        N = 0
        simrange = tqdm(U) if show_progress else U
        for u in simrange:
            if len(Time) == 1:
                dT = [Time[0], Time[0]+ts]
            else:
                dT = [Time[N], Time[N + 1]]
            if self.nu == 1:
                u = u[0]
            x, reward, terminated, truncated, info = self.gym_env.step(u) # this line is the only way this this self.simulate() differs from super.simulate().
            X.append(x.ravel())
            N += 1
            if N == nsim or terminated or truncated:
                break
        Yout = np.asarray(X).reshape(nsim+1, -1)
        Uout = U.reshape(nsim, -1)
        return {'Y': Yout, 'U': Uout, 'X': np.asarray(X)}

    def get_x0(self):
        x, info = self.gym_env.reset()
        return x

    def get_U(self, nsim=None):
        if nsim is None:
            nsim = self.nsim
        U = np.empty((nsim, self.nu),dtype=self.gym_env.action_space.dtype)
        for n in range(nsim):
            U[n,:] = self.gym_env.action_space.sample().ravel()
        return U

    def equations(self, x, t, u):
        return None
