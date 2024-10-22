from scipy.io import loadmat
from gym import spaces, Env

import numpy as np
from neuromancer.psl.building_envelope import BuildingEnvelope, systems


class BuildingEnv(Env):
    """Custom Gym Environment for simulating building energy systems.

    This environment models the dynamics of a building's thermal system, 
    allowing for control actions to be taken and observing the resulting 
    thermal comfort levels. The environment adheres to the OpenAI Gym 
    interface, providing methods for stepping through the simulation, 
    resetting the state, and rendering the environment.

    Attributes:
        metadata (dict): Information about the rendering modes available.
        ymin (float): Minimum threshold for thermal comfort.
        ymax (float): Maximum threshold for thermal comfort.
    """

    def __init__(self, simulator, seed=None, fully_observable=True):
        super().__init__()
        if isinstance(simulator, BuildingEnvelope):
            self.model = simulator
        else:
            self.model = systems[simulator](seed=seed)
        self.action_space = spaces.Box(
            self.model.umin, self.model.umax, shape=self.model.umin.shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self.model.x0.shape, dtype=np.float32)
        self.fully_observable = fully_observable
        self.reset()

    def step(self, action):
        u = np.asarray(action)
        d = self.model.get_D(1).flatten()
        # model should accept either 1D arrays or 2D (n, 1) arrays
        self.x, self.y = self.model(self.x, u, d)
        self.t += 1
        self.X_rec = np.append(self.X_rec, self.x)
        reward = self.reward(u, self.y)
        done = self.t == self.model.nsim
        return self.obs, reward, done, dict(X_rec=self.X_rec)
    
    def reward(self, u, y, ymin=21.0, ymax=23.0):
        # energy minimization
        action_loss = 0.1 * np.sum(u > 0.0)

        # thermal comfort 
        inbound_reward = 5. * np.sum((ymin < y) & (y < ymax))

        return inbound_reward - action_loss
    
    @property
    def obs(self):
        return (self.y, self.x)[self.fully_observable].astype(np.float32)

    def reset(self):
        self.t = 0
        self.x = self.model.x0
        self.y = self.model.C * self.x
        self.X_rec = np.empty(shape=[0, 4])
        return self.obs

    def render(self, mode='human'):
        pass
