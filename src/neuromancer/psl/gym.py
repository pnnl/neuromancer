import numpy as np
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

    def __init__(self, simulator, seed=None, fully_observable=True, backend='numpy'):
        super().__init__()
        if isinstance(simulator, BuildingEnvelope):
            self.model = simulator
        else:
            self.model = systems[simulator](seed=seed, backend=backend)
        self.action_space = spaces.Box(
            self.model.umin, self.model.umax, shape=self.model.umin.shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self.model.x0.shape, dtype=np.float32)
        self.fully_observable = fully_observable
        self.reset(seed=seed)

    def step(self, action):
        u = np.asarray(action)
        d = self.model.get_D(1).flatten()
        # expect the model to accept both 1D arrays and 2D arrays
        self.x, self.y = self.model(self.x, u, d)
        self.t += 1
        self.X_rec = np.append(self.X_rec, self.x)
        reward = self.reward(u, self.y)
        done = self.t == self.model.nsim
        truncated = False
        return self.obs, reward, done, truncated, dict(X_rec=self.X_rec)
    
    def reward(self, u, y, ymin=20.0, ymax=22.0):
        # energy minimization
        # u[0] is the nominal mass flow rate, u[1] is the temperature difference
        q = self.model.get_q(u).sum()  # q is the heat flow in W
        k = np.sum(u != 0.0)  # number of actions
        action_loss = 0.01 * q + 0.01 * k

        # thermal comfort
        comfort_reward = 5. * np.sum((ymin < y) & (y < ymax))  # y in °C

        return comfort_reward - action_loss
    
    @property
    def obs(self):
        return (self.y, self.x)[self.fully_observable]

    def reset(self, seed=None, options=None):
        seed_everything(seed)
        self.t = 0
        self.x = self.model.x0
        self.y = self.model.C * self.x
        self.X_rec = np.empty(shape=[0, 4])
        return self.obs, dict(X_rec=self.X_rec)

    def render(self, mode='human'):
        pass


# allow the custom envs to be directly instantiated by gym.make(env_id)
for env_id in systems:
    register(
        env_id,
        entry_point='neuromancer.psl.gym:BuildingEnv',
        kwargs=dict(simulator=env_id),
    )