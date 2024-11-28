import numpy as np
import torch
from gymnasium import spaces, Env
from gymnasium.envs.registration import register
from neuromancer.utils import seed_everything
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

    def __init__(self, simulator, seed=None, fully_observable=False, 
                 ymin=20.0, ymax=22.0, backend='numpy'):
        super().__init__()
        if isinstance(simulator, BuildingEnvelope):
            self.model = simulator
        else:
            self.model = systems[simulator](seed=seed, backend=backend)
        self.fully_observable = fully_observable
        self.ymin = ymin
        self.ymax = ymax
        obs, _ = self.reset(seed=seed)
        self.action_space = spaces.Box(
            self.model.umin, self.model.umax, shape=self.model.umin.shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=[len(obs)], dtype=np.float32)

    def step(self, action):
        u = np.asarray(action)
        self.d = self.get_disturbance()
        # expect the model to accept both 1D arrays and 2D arrays
        self.x, self.y = self.model(self.x, u, self.d)
        self.t += 1
        self.X_rec = np.append(self.X_rec, self.x)
        obs = self.get_obs()
        reward = self.get_reward(u, self.y)
        done = self.t == self.model.nsim
        truncated = False
        return obs, reward, done, truncated, dict(X_rec=self.X_rec)
    
    def get_reward(self, u, y, ymin=20.0, ymax=22.0):
        # energy minimization
        # u[0] is the nominal mass flow rate, u[1] is the temperature difference
        q = self.model.get_q(u).sum()  # q is the heat flow in W
        k = np.sum(u != 0.0)  # number of actions
        action_loss = 0.01 * q + 0.01 * k

        # thermal comfort
        comfort_reward = 5. * np.sum((ymin < y) & (y < ymax))  # y in °C

        return comfort_reward - action_loss
    
    def get_disturbance(self):
        return self.model.get_D(1).flatten()
    
    def get_obs(self):
        obs_mask = torch.as_tensor(self.model.C.flatten(), dtype=torch.bool)
        self.y = self.x[obs_mask]
        d = self.d if self.fully_observable else self.d[self.model.d_idx]
        obs = self.x if self.fully_observable else self.y
        obs = np.hstack([obs, self.ymin, self.ymax, d])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        seed_everything(seed)
        self.t = 0
        self.x = self.model.x0
        self.d = self.get_disturbance()
        self.X_rec = np.empty(shape=[0, 4])
        return self.get_obs(), dict(X_rec=self.X_rec)

    def render(self, mode='human'):
        pass


# allow the custom envs to be directly instantiated by gym.make(env_id)
for env_id in systems:
    register(
        env_id,
        entry_point='neuromancer.psl.gym:BuildingEnv',
        kwargs=dict(simulator=env_id),
    )