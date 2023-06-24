
'''

dependencies:
psl
gymnasium
matplotlib
skrl
wandb
'''

import gymnasium
import json
import numpy as np
from numpy import dtype
import os
from typing import Union, Sequence, Any

from neuromancer.psl.nonautonomous import systems

class PslGym(gymnasium.Env):
   def __init__(
         self,
         env_str: str,
         reward_class: 'Reward',
         reward_norm: Union[str, float]='mm',
         action_constraints=None,
         state_constraints=None,
         max_u_delta: Union[float, np.ndarray]=None,
         max_x_delta: Union[float, np.ndarray]=None,
         reset_freq: int=1,
         ref_dims: np.array=None, # dtype=np.int32
         xy='y',
         dtype=np.float32,
         seed=None
      ):
      """
      Wraps a psl.nonautonomous.ODE_NonAutonomous system in a Gymnasium environment for reinforcement learning.
      Inputs:
      :param env_str: class name of psl.nonautonomous.ODE_NonAutonomous subclass (str) 
         values: ['Autoignition', 'Brusselator1D', 'ChuaCircuit', 'DoublePendulum', 'Duffing', 'Lorenz96', 'LorenzSystem', 'LotkaVolterra', 'Pendulum', 'RosslerAttractor', 'ThomasAttractor', 'UniversalOscillator', 'VanDerPol', 'Actuator', 'CSTR', 'DuffingControl', 'HindmarshRose', 'InvPendulum', 'IverSimple', 'LorenzControl', 'SEIR_population', 'SwingEquation', 'Tank', 'ThomasAttractorControl', 'TwoTank', 'VanDerPolControl', 'SimpleSingleZone', 'Reno_full', 'Reno_ROM40', 'RenoLight_full', 'RenoLight_ROM40', 'Old_full', 'Old_ROM40', 'HollandschHuys_full', 'HollandschHuys_ROM100', 'Infrax_full', 'Infrax_ROM100', 'LinearSimpleSingleZone', 'LinearReno_full', 'LinearReno_ROM40', 'LinearRenoLight_full', 'LinearRenoLight_ROM40', 'LinearOld_full', 'LinearOld_ROM40', 'LinearHollandschHuys_full', 'LinearHollandschHuys_ROM100', 'LinearInfrax_full', 'LinearInfrax_ROM100']

      :param reward_class:
         fields: min, max (float)
         methods: 
            reward(
               x (numpy array, current state of same dimensions os ODE_NonAutonomous.X0), 
               ref (numpy array, target reference state, same dimensions as x), 
               num_violations (float, measure of penalty for constraint violations. 0=no penalty.) 
               d (float, a hyperparameter.)
               )
               output: float metric goodness measure of x (greater is better)

      :param reward_norm: (str, 'mm' or float, normalization method for reward. If 'mm', penalties are component-wise divided by their range, 
      otherwise, penalties are divided by reward_norm.
      
      :param action_constraints: function (
         action, (numpy array, corresponding to single timestep ODE_NonAutonomous.U)
         a_prev, (numpy array, action from previous timestep)
         umin, (numpy array, lower bound on action)
         umax, (numpy array, upper bound on action)
         max_a_delta (numpy array or float, represents maximum difference between action and a_prev (either componentwise or congregated(
         )
         output: 
            action: (numpy array, same as input action but possibly truncated)
            num_violations: (float, penalty measure for input, greater is worse)

      :param state_constraints: function (
         state: (numpy array, corresponding to single timestep ODE_NonAutonomous.X, concatenated with reference target)
         state_prev: (numpy array, state from previous timestep)
         xmin: (numpy array, lower bound on state)
         xmax: (numpy array, upper bound on state)
         max_x_delta: numpy array or float, represents maximum absolute difference between state and state_prev
         )
         output:
            penalty: float, penalty measure for current state
      :param reset_freq: number of calls to self.reset() before reference trajectory is reset (int)
      :param ref_dims: (numpy array of ints, indexes of x to be measured againt ref_t. Other x dims are ignored in reward class. If None, all dims used in computing reward. (numpy array))
      :param xy: (str, 'x' or 'y', whether to use full state (X) or observation (Y) in corresponding psl system for self.s)

      Fields:

      :field s: (numpy array, s is for state, it includes psl's X and the reference target.)

      """

      super(PslGym, self).__init__()

      self.action_constraints = action_constraints if action_constraints is not None else lambda a, a_prev, u_min, u_max, max_u_delta: (a, 0)
      self.state_constraints = state_constraints if state_constraints is not None else lambda s, s_prev, x_min, x_max, max_x_delta: (s, 0)

      for key in ['xy','max_u_delta','max_x_delta','reset_freq','ref_dims', 'env_str', 'dtype', 'seed']:
         setattr(self, key, locals()[key])

      for key in ['s','s_prev','a','a_prev','x_min','x_max','u_min','u_max']:
         setattr(self, key, None)

      self.render_mode = 'human' # FIXME: add render function to display animated environment

      self.t = 0 # time index
      self.resets = 0 # track how many times self.reset() has been called
      self.step_nsim = 1 # number of psl steps to take per call to step(), gets incremented if fails

      self.psl_env = systems[self.env_str]()
      self.nsim = self.psl_env.nsim

      self.nx, self.ns, self.na, self.u_stats, self.u_min, self.u_max, self.x_stats, self.x_min, self.x_max, self.reward_norm = \
         self.make_bounds() 

      self.reward_class = reward_class(norm = (self.x_stats['max']-self.x_stats['min']) if reward_norm=='mm' else reward_norm)

      self.ref_dims = ref_dims if ref_dims is not None else np.arange(0, self.nx, dtype=int)

      self.observation_space = BoxSpace(low=np.concatenate( (self.x_min, self.x_min)), high=np.concatenate((self.x_max, self.x_max)), dtype=self.dtype, shape=(self.ns,)) # include targets in observation
      self.action_space = BoxSpace(low=self.u_min, high=self.u_max, dtype=self.dtype, shape=(self.na,))

   def render(self):
      assert self.render_mode == 'human', 'only human render mode supported'
      # FIXME see twotank_game.py for example of how to render using pygame

   def get_U(self, nsim):
      """
      signals.signals.keys()
         dict_keys(['sin', 'square', 'sawtooth', 'walk', 'noise', 'step', 'spline', 'sines', 'arma', 'prbs'])
      """
      # FIXME vary signal type for U sampling or manually sample U
      # sig = inspect.signature (self.psl_env.get_U)
      # if 'type' in sig.parameters.keys():
      # U = self.psl_env.get_U(nsim=self.psl_env.nsim,type='step')
      # signals = neuromancer.signals.keys()

      return self.psl_env.get_U(nsim=nsim)

   def set_ref_t(self, seed=None, type='walk'):
      '''
      gets random reference trajectory. If step samples a step function, else sample random simulation self.psl_env
      no return, updates self.ref_t
      # FIXME: add support for seed, will come with rng pull request
      # FIXME: add support for type for building envelope model
      '''
      # if seed is not None:
      #    self.psl_env.rng.save_random_state()
      #    self.psl_env.rng.set_seed(seed)

      U = self.get_U(nsim=self.nsim)
      x0 = self.psl_env.get_x0()
      sim = self.psl_env.simulate(U=U, x0=x0)

      # if seed is not None:
      #    self.psl_env.rng.restore_random_state()

      self.ref_t = np.concatenate((x0.reshape(1,-1),sim['X']),axis=0) if self.xy=='x' else np.concatenate((sim['Y'][0:1,:],sim['Y']),axis=0)
      self.trace = np.zeros(self.ref_t.shape)
      self.rewards = np.zeros((self.trace.shape[0],1))
      self.rewards[0] = 1
      return x0, self.ref_t

   def get_trace(self):
      return self.trace[:self.t+1,:], self.rewards[:self.t+1]

   def make_bounds(self):
      """
      Get statistics from psl for constructing observation and action spaces
      """
      # nx = self.psl_env.nx if hasattr(self.psl_env, 'nx') else 1
      na = self.psl_env.nu if hasattr(self.psl_env, 'nu') else 1
      u_stats = self.psl_env.stats['U']
      u_min = self.psl_env.umin if hasattr(self.psl_env, 'umin') else u_stats['min'].ravel()
      u_max = self.psl_env.umax if hasattr(self.psl_env, 'umax') else u_stats['max'].ravel()
      u_stats['min'], u_stats['max'] = u_min, u_max
      x_stats = self.psl_env.stats['X'] if self.xy=='x' else self.psl_env.stats['Y']
      x_min, x_max  = x_stats['min'].ravel(), x_stats['max'].ravel()
      nx = x_min.ravel().shape[0]
      ns = 2*nx

      reward_norm = x_max - x_min

      return nx, ns, na, u_stats, u_min, u_max, x_stats, x_min, x_max, reward_norm


   def aggregate_bounds(self, old, new, key):
      old, new = np.array(old).ravel(), np.array(new).ravel()
      return np.min([old,new]) if key=='min' else np.max([old,new])

   def synch_bounds_file(self, u_min, u_max, x_min, x_max, directory=None):
      '''
      Create bounds file if it doesn't exist, otherwise update it with new bounds
      directory is pslgym_bounds in current working directory
      '''
      directory = os.getcwd() if directory is None else directory
      bounds_file_dir = os.path.join(directory, 'pslgym_bounds')

      if not os.path.exists(bounds_file_dir):
         os.mkdir(bounds_file_dir)

      bounds_file = os.path.join(bounds_file_dir, self.env_str+'_bounds.json')
      bounds_file = os.path.abspath(bounds_file)

      if os.path.exists(bounds_file):
         with open(bounds_file,'r') as fi:
            old_bounds = json.load(fi)
            for key in ['u_min','u_max','x_min','x_max']:
               locals()[key] = self.aggregate_bounds(old_bounds[key], locals()[key], key) # locals()[key] is arg: u_min, ..., x_max

      bounds = dict()
      for key in ['u_min','u_max','x_min','x_max']:
         bounds[key] = locals()[key].ravel().tolist()

      with open(bounds_file,'w') as fi:
         json.dump(bounds, fi)

      return bounds, bounds_file


   def set_s(self, sim):
      """
      Set current state self.s from simulation, and self.prev_s
      also check if min/max changed
      """

      self.X = sim['X'][-1]
      x = sim['X'][-1] if self.xy=='x' else sim['Y'][-1]
      x = np.array(x).ravel()
      self.trace[self.t,:] = x

      min_max_changed = False
      nan = np.any(np.isnan(x))
      if nan:
         min_max_changed = None
      else:
         less_idxs = np.where(x < self.x_min)
         if less_idxs[0].size > 0:
            self.x_min[less_idxs] = x[less_idxs]
            min_max_changed = True

         greater_idxs = np.where(x > self.x_max)
         if greater_idxs[0].size > 0:
            self.x_max[greater_idxs] = x[greater_idxs]
            min_max_changed = True
      
      self.s_prev = self.s
      self.s = np.concatenate((x,self.ref_t[self.t,:]),axis=0)
      info = { \
         'min_max_changed' : min_max_changed,
         'outside_bounds' : not self.observation_space.contains(self.s),
         'state_violation' : self.state_constraints(self.s, self.s_prev, self.x_min, self.x_max, self.max_x_delta),
         'nan' : nan}
      return self.s, info


   def set_a(self, a):
      """
      Similar to self.set_s(), but for action
      """
      a = np.array(a).ravel()
      min_max_changed = False

      nan = np.any(np.isnan(a))
      if nan:
         min_max_changed = None
      else:
         less_idxs = np.where(a < self.u_min)
         if less_idxs[0].size > 0:
            self.u_min[less_idxs] = a[less_idxs]
            min_max_changed = True

         greater_idxs = np.where(a > self.u_max)
         if greater_idxs[0].size > 0:
            self.u_max[greater_idxs] = a[greater_idxs]
            min_max_changed = True

      self.a_prev = self.a
      self.a = a
      info = { \
         'min_max_changed' : min_max_changed,
         'outside_bounds' : not self.action_space.contains(self.a),
         'action_violation' : self.action_constraints(self.a, self.a_prev, self.u_min, self.u_max, self.max_u_delta),
         'nan' : nan }
      return self.a, info

   def reset(self, seed: 'int | None' = None, options: 'dict[str, Any] | None' = None):
      """
      Start a new episode (must be called once before step())
      """
      self.t = 0
      print('pslgym resets',self.resets)
      if self.resets % self.reset_freq==0:
         typ = 'walk' # FIXME
         x0, _ = self.set_ref_t(seed=seed, type=typ)

      self.resets = self.resets + 1

      if seed is not None:
         self.psl_env.rng.save_random_state()
         self.psl_env.rng.set_seed(seed)

      sim = self.try_sim_step(a=None, x0=x0)

      if seed is not None:
         self.psl_env.rng.restore_random_state()

      s, s_info = self.set_s(sim)
      if s_info['min_max_changed']:
         self.synch_bounds_file(self.u_min, self.u_max, self.x_min, self.x_max)

      info = { 'current time':self.t, 'num resets':self.resets, 's_info':s_info }
      return s, info

   def replicate_action(self, a, n=2):
      """
      Since PSL doesn't support a single control action step
      """
      U=a.reshape(1,self.na)
      U = np.array([U for i in range(n)]).squeeze(1)
      return U
   
   def try_sim_step(self, a=None, x0=None):
      """
      Run a single simulation step from self.psl_env
      """
      U = self.replicate_action(a) if a is not None else self.psl_env.get_U(nsim=2)
      x0 = self.psl_env.get_x0() if x0 is None else x0
      sim = self.psl_env.simulate(U=U, x0=x0, nsim=1)
      return sim

   def step(self, action):
      '''
      Required Gymnasium Env method, run a single step of the simulation
      returns the observation (tuple), reward (float), terminated (bool), info (dict ['current time':self.t, 'action violation':None, 'got_NaN':False])
      '''
      # FIXME: optionally denormalize action
      self.t = self.t + 1

      a, u_info = self.set_a(action)

      sim = self.try_sim_step(a=action, x0=self.X)

      s, s_info = self.set_s(sim)

      if u_info['min_max_changed'] or s_info['min_max_changed']:
         self.synch_bounds_file(self.u_min, self.u_max, self.x_min, self.x_max)

      truncated = False
      got_NaN = u_info['nan'] or s_info['nan']
      if got_NaN:
         reward = self.reward_class.min # max penalty for NaN
         print('Got NaN, reward',reward)
         print(self.s_prev, self.a_prev,'->', self.s, self.a)
         truncated = True
      else:
         reward = self.reward_class.reward( x=self.s[:self.nx], ref=self.ref_t[self.t-1].ravel(), ref_dims=self.ref_dims, norm=self.reward_norm, d=1 )
         self.rewards[self.t] = reward
         # check if reward is scalar
         if not np.isscalar(reward):
            print('reward is not scalar, self.t',self.t)
            breakpoint()

      terminated = self.t>=self.nsim-1
      if terminated:
         print('episode complete') 

      info = { 'time':self.t,
               'reset':self.resets,
               'got_NaN':got_NaN,
               'u_info':u_info,
               's_info':s_info}

      # FIXME: optionally normalize state
      return s, reward, terminated, truncated, info

# self.observation_space.contains = lambda s: self.observation_space.super().contains(np.array(s,dtype=self.dtype).ravel())
class BoxSpace(gymnasium.spaces.Box):
   """
   Wrapper for Box space, more robust contains() method
   """
   def __init__(self, low, high, shape=None, dtype=np.float32):
      super().__init__(low=low, high=high, shape=shape, dtype=dtype)
   def contains(self, x):
      return super().contains(np.array(x,dtype=self.dtype).ravel())

class BallSpace(gymnasium.spaces.Space):
   """
   Like a BoxSpace, but with a ball
   WARNING: untested
   """
   def __init__(self, radius: float, nx: int, shape: Sequence[int] | None = None, n_balls: int = 2, dtype: dtype | None = None, seed: int | np.random.Generator | None = None):
      super().__init__(shape=shape, dtype=dtype, seed=seed)
      self.radius = radius
      self.is_np_flattenable = True
      self.shape = shape
      self.dtype = dtype
      self.seed(seed)
      self.low = low
      self.high = high
      self.nx = nx
      self.rng = np.random.default_rng(seed)

   def sample(self) -> np.array:
      # FIXME: generalize to nx dimensions and n balls, spherical coordinates
      r = self.rng.uniform(low=0, high=self.radius, size=self.nx)
      theta = self.rng.uniform(low=0, high=2*np.pi, size=self.nx)
      x1 = r * np.cos(theta)
      y1 = r * np.sin(theta)
      r = self.rng.uniform(low=0, high=self.radius, size=self.nx)
      theta = self.rng.uniform(low=0, high=2*np.pi, size=self.nx)
      x2 = r * np.cos(theta)
      y2 = r * np.sin(theta)
      return np.array([x1, y1, x2, y2])

   def contains(self, x: Any) -> bool:
      return True if np.sum(x**2) <= self.radius**2 else False

   def seed(self, seed: int | None = None) -> list[int]:
      self.rng = np.random.default_rng(seed)

class Reward():
   """docstring for Reward"""
   def __init__(self, norm: Union[np.ndarray, float], min_r: float, max_r: float):
      self.norm = norm.reshape(1,-1) if isinstance(norm,np.ndarray) else norm
      self.min, self.max = min_r, max_r # min/max are guidelines for reward, not hard limits
   def reward(self,x: np.array,ref: np.array,ref_dims,norm,d: float=1)->float:
      raise NotImplementedError

class MSE(Reward):
   """MSE reward function"""
   def __init__(self, norm, min_r=-3, max_r=1):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r)
   def reward(self,x,ref,ref_dims,norm,d=1):
      err = (x[ref_dims]-ref[ref_dims])**2
      err = err / self.norm
      err = err.ravel()
      err = np.mean(err)
      r = d - err
      return r

class MAE(Reward):
   """Mean Absolute Error reward function"""
   def __init__(self, norm, min_r=-3, max_r=1):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r)
   def reward(self,x,ref,ref_dims,norm,d=1):
      err = np.abs(x[ref_dims]-ref[ref_dims])
      err = err / self.norm
      reward = d - np.mean( err )
      return reward
