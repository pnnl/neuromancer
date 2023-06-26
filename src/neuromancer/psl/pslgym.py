'''

Wraps PSL Nonautonomous systems in Gymnasium environment for reinforcement learning

Dependencies:
   psl
   gymnasium
   matplotlib
   skrl
   wandb

Authors:
   Seth Briney
   Lyric Otto

'''

import gymnasium
import json
import numpy as np
from numpy import dtype
import os
from typing import Union, Sequence, Any
import warnings

from neuromancer.psl.nonautonomous import systems

class PslGym(gymnasium.Env):
   def __init__(
         self,
         psl_sys: str,
         reward_instance: 'Reward', # see bottom of this file
         reset_freq: int=1,
         do_trace: bool=False,
         render_mode: str=None,
         dtype: dtype=np.float32,
         seed: Union[int,'RNG()']=None,
         p: bool=True,
         d: bool=False
      ):
      """
      Wraps a psl.nonautonomous.ODE_NonAutonomous system in a Gymnasium environment for reinforcement learning.

      Parameters:

      :param psl_sys: class name of psl.nonautonomous.ODE_NonAutonomous subclass (str) 

      :param reward_instance: (if class not instance is passed instanitates with defaults)
         required fields: norm: Union[np.array, float]
         suggested fields: norm: Union[np.array, float], min, max (float)
         methods: 
            reward(y, ref)
               -> float metric goodness measure of obs (greater is better)

      :param reset_freq: (int) number of episodes (calls to self.reset()) before reference trajectory is changed

      :param do_trace: (bool) if true keep trace of self.u_trace, self.y_trace and self.rewards per-episode


      Fields:

      :field t: (int), current timestep (per episode)
      :field y: (numpy array), observation sim['Y'] from psl_sys
      :fields (y/u)_(min/max): (numpy array) soft min/max values
      :field refs: (numpy array), targets for y, indexed by self.t
      :field u: (numpy array), control action like psl_sys.get_U()
      :field psl_env: (str), psl.nonautonomous.ODE_NonAutonomous system named by psl_sys string.
      :field observation_space: (gymnasium.spaces), includes y and refs[t]
      :field render_mode: (str), FIXME - see twotank_game.py
      :field p: (str), flag for printing certain things
      :field d: (str), flag for debug prints

      # Optional fields (if do_trace)
         u_trace, y_trace, rewards

      # Indexing alignment notes:
         refs[t] is the target for the current y==y_trace[t]
         y_trace.shape[0] ==  u_trace.shape[0] (final y has no corresponding action)
         y_trace.shape[0] ==     refs.shape[0] (initial y had no target)

         rewards[t] is the reward associated with taking action:
            u==u_trace[t] (the action taken immediately after observing)
            y==y_trace[t] (with target)
            refs[t]

      obs=concatenate(y,refs[t]) is an element of observation_space
      obs is returned from reset() and step(), and only constructed before being returned using get_obs()

      All values are assumed to have physical units, any desired (de)normalization takes place externally.
      y_min, y_max, u_min, u_max, psl_env.stats['Y'], psl_env.stats['U'] can all be used for normalization.

      TODO:

         convert signals to use rng object and uncomment corresponding statements
         figure out more robust way to handle max/min bounds for action/observation spaces
            ideally:
               (min/max)_u should be exact range of allowed actions
               (min/max)_y should at least contain true bounds

      """

      super(PslGym, self).__init__()

      self.psl_env = systems[psl_sys]()
      self.nsim = self.psl_env.nsim-1
      if p:
         print('PslGym init, psl system:',psl_sys,'self.nsim:',self.nsim,'self.psl_env.nsim',self.psl_env.nsim)

      self.t = 0 # timestep index
      self.episodes = 0 # track how many times self.reset() has been called

      # Set most of the attributes here
      for key in ['psl_sys','reward_instance','reset_freq','do_trace','render_mode','dtype','seed','p','d']:
         setattr(self, key, locals()[key])
      for key in ['y','u','y_min','y_max','u_min','u_max']:
         setattr(self, key, None)

      # Construct bounds, reward, and obs/act spaces
      self.nu, self.ny, self.u_min, self.u_max, self.y_min, self.y_max = \
         self.get_bounds()

      # allocate memory for trace values if appropriate
      self.refs =       np.empty((self.nsim, self.ny), dtype=dtype) # reference targets
      if p:
         print('self.refs.shape',self.refs.shape)
      if do_trace: # extra baggage if not used, nice for visualizing and computing dev metrics
         self.y_trace = np.empty((self.nsim, self.ny), dtype=dtype)
         self.u_trace = np.empty((self.nsim, self.nu), dtype=dtype)
         self.rewards = np.empty((self.nsim, 1),       dtype=dtype)
         if p:
            print('shapes: self.y_trace',self.y_trace.shape,'self.u_trace',self.u_trace.shape,'self.rewards',self.rewards.shape)

      # Construct obs/act space:
      self.observation_space = BoxSpace(low=np.concatenate( (self.y_min, self.y_min)), \
                                        high=np.concatenate((self.y_max, self.y_max)), \
                                        dtype=self.dtype, shape=(2*self.ny,)) # include reference targets in observation

      self.action_space = BoxSpace(low=self.u_min, high=self.u_max, dtype=self.dtype, shape=(self.nu,))

      # Define reward function
      reward_norm = (self.y_max-self.y_min)
      if p:
         print('\nreward_norm\n',reward_norm)
      if not isinstance(self.reward_instance, Reward):
         self.reward_instance = self.reward_instance(norm=reward_norm)
      else:
         self.reward_instance.set_norm(reward_norm)


   def render(self):
      """
      Visualize y, refs, rewards (maybe actions)
      FIXME see twotank_game.py for example of how to render using pygame
      should have the option to save a plot especially at the end of an episode, and save plot in reset().
      """
      pass

   def get_U(self, nsim):
      """
      returns control actions

      signals.signals.keys()
      FIXME vary signal type for U sampling or manually sample U
      sig = inspect.signature (self.psl_env.get_U)
      if 'type' in sig.parameters.keys():
      U = self.psl_env.get_U(nsim=self.psl_env.nsim,type='step')
      signals = neuromancer.signals.keys()
      """

      U = self.psl_env.get_U(nsim=nsim)
      if self.p: print('get_U, nsim:',nsim,'U.shape',U.shape)

      return U

   def set_refs(self, seed=None, type='walk')->np.array:
      '''
      get random reference trajectory from psl_env.simulate
      set self.refs
      returns initial condition

      # FIXME: add support for seed, will come with rng pull request
      # FIXME: add support for type for building envelope model
      '''
      # if seed is not None:
      #    self.psl_env.rng.save_random_state()
      #    self.psl_env.rng.set_seed(seed)

      nsim = self.psl_env.nsim
      U   = self.get_U(nsim=nsim)
      x0  = self.psl_env.get_x0()
      sim = self.psl_env.simulate(U=U, x0=x0)

      # if seed is not None:
      #    self.psl_env.rng.restore_random_state()

      self.refs[:,:] = np.array(sim['Y'], dtype=self.dtype).reshape(self.refs.shape)

      if self.p: print('set_refs','nsim',nsim,'U.shape',U.shape,'sim[\'Y\'].shape',sim['Y'].shape)

      return x0

   def set_do_trace(self,do_trace: bool=True):
      '''
      WARNING: must call reset() immediately after set_do_trace()
      '''
      self.do_trace = do_trace

   def mae(self,ys, refs):
      assert self.do_trace, 'mae only works if self.do_trace'
      if ys.shape[0]==0:
         return None
      err = ys-refs
      mae = float(np.mean( np.sum( np.abs( err ),axis=1).ravel() ) )
      return mae

   def get_trace(self):
      '''
      Note we can't include self.t because we don't know r yet
      '''
      assert self.do_trace, 'get_trace only works if self.do_trace'
      u_trace = self.u_trace[:self.t,:]
      refs    =    self.refs[:self.t,:]
      rewards = self.rewards[:self.t,:]
      y_trace = self.y_trace[:self.t,:] # include initial obs
      mae = self.mae(ys=y_trace[1:,:], refs=refs[:-1,:])
      return mae, u_trace, y_trace, self.refs, rewards

   def get_bounds(self):
      """
      Get statistics from psl for constructing observation and action spaces
      """
      u_min = np.array(self.psl_env.umin if hasattr(self.psl_env, 'umin') else psl_env.stats['U']['min'], dtype=self.dtype).ravel()
      u_max = np.array(self.psl_env.umax if hasattr(self.psl_env, 'umax') else psl_env.stats['U']['max'], dtype=self.dtype).ravel()
      y_min = np.array(self.psl_env.ymin if hasattr(self.psl_env, 'ymin') else self.psl_env.stats['Y']['min'], dtype=self.dtype).ravel()
      y_max = np.array(self.psl_env.ymax if hasattr(self.psl_env, 'ymax') else self.psl_env.stats['Y']['max'], dtype=self.dtype).ravel()

      nu = int(u_min.ravel().shape[0])
      ny = int(y_min.ravel().shape[0])

      if self.p: print('u_min',u_min,'u_max',u_max,'y_min',y_min,'y_max',y_max)

      return nu, ny, u_min, u_max, y_min, y_max

   def set_y(self, sim):
      """
      Set current observation self.obs from simulation
      also check if min/max changed
      """

      self.X = sim['X'][-1] # store psl state as initial condition
      self.y = np.array(sim['Y'][-1], dtype=self.dtype).ravel()

      if self.do_trace:
         self.y_trace[self.t,:] = self.y

      return self.y

   def set_u(self, u):
      """
      Similar to self.set_y(), but for control actions
      """
      self.u = np.array(u, dtype=self.dtype).ravel()
      if self.do_trace:
         self.u_trace[self.t,:] = u
      
      return u

   def get_reward(self):
      reward = self.reward_instance.reward(y=self.y, ref=self.refs[self.t-1])
      if self.do_trace:
         self.rewards[self.t-1,:] = reward
      return reward

   def get_obs(self):
      ref = self.refs[self.t]
      return np.concatenate( (self.y, ref) )

   def reset(self, seed: int=None):
      """
      Start a new episode (must be called once before step())
      """
      self.t = 0

      if self.episodes % self.reset_freq==0:
         if self.p: print('pslgym reset, new refs')
         self.x0 = self.set_refs(seed=seed)

      self.episodes = self.episodes + 1

      if self.p: print('pslgym reset, episode',self.episodes)

      # if seed is not None:
      #    self.psl_env.rng.save_random_state()
      #    self.psl_env.rng.set_seed(seed)

      sim = self.try_sim_step(u=None, x0=self.x0)
      self.set_y(sim)

      # if seed is not None:
      #    self.psl_env.rng.restore_random_state()

      info = {'current_time':self.t, 'num_resets':self.episodes}
      return self.get_obs(), info

   def replicate_action(self, u, n=2):
      """
      Since PSL doesn't support a single control action step
      """
      U = u.reshape(1,self.nu)
      U = np.array([U for i in range(n)]).squeeze(1)
      return U
   
   def try_sim_step(self, u=None, x0=None):
      """
      Run a single simulation step from self.psl_env
      """
      U = self.replicate_action(u) if u is not None else self.psl_env.get_U(nsim=2)
      x0 = x0 if x0 is not None else self.psl_env.get_x0()
      sim = self.psl_env.simulate(U=U, x0=x0, nsim=1)
      return sim

   def step(self, action):
      '''
      Required Gymnasium Env method, run a single step of the simulation
      returns the observation (tuple), reward (float), terminated (bool), info (dict ['current time':self.t, 'action violation':None, 'got_NaN':False])
      '''
      self.set_u(action)

      truncated = False
      try:
         sim = self.try_sim_step(u=self.u, x0=self.X)
         if self.d: print('step u',self.u,'t',self.t,end=' ')
         self.t = self.t + 1
      except:
         print(f'simulation failed, time {self.t}/{self.nsim}, episode {self.episodes}')
         truncated = True

      self.set_y(sim)
      reward = self.get_reward()
      if self.d: print('y',self.y,'r',reward,f't {self.t}/{self.nsim-1}')

      terminated = (self.t==self.nsim-1)
      if self.p and terminated: print(f'step terminated (episode complete) t {self.t}/{self.nsim-1}')

      obs = self.get_obs()
      if self.d: print('obs',obs,f't {self.t}/{self.nsim-1}')

      info = { 'time':self.t, 'episodes':self.episodes }
      return obs, reward, terminated, truncated, info



# Gymnasium Spaces:

class BoxSpace(gymnasium.spaces.Box):
   """
   Wrapper for Box space, more robust contains() method
   """
   def __init__(self, low, high, shape=None, dtype=np.float32):
      super().__init__(low=low, high=high, shape=shape, dtype=dtype)
      self.ny = self.shape[0]//2
   def contains(self, x):
      return super().contains(np.array(x,dtype=self.dtype).ravel())

class BallSpace(gymnasium.spaces.Space):
   """
   Like a BoxSpace, but with a ball
   WARNING: untested
   """
   def __init__(self, radius: float, nx: int, shape: Union[Sequence[int], None]=None, n_balls: int = 2, dtype: dtype=np.float32, seed: Union[int, np.random.Generator, None]=None):
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


# Reward Classes:

class Reward():
   """docstring for Reward"""
   def __init__(self, norm: Union[np.ndarray, float], min_r: float, max_r: float, dtype: dtype=np.float32):
      self.norm = norm.reshape(1,-1) if isinstance(norm,np.ndarray) else norm
      self.min, self.max = min_r, max_r # min/max are guidelines for reward, not hard limits
      self.dtype=dtype
   def set_norm(self, norm):
      self.norm = norm
   def reward(self, y: np.array, ref: np.array, norm: Union[float, np.array], d: float=1)->float:
      raise NotImplementedError

class MSE(Reward):
   """MSE reward function"""
   def __init__(self, norm=None, min_r=-3, max_r=1, dtype=np.float32):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r, dtype=dtype)
   def reward(self, y, ref):
      err = (y-ref)**2
      err = err if self.norm is None else err/self.norm
      err = err.ravel()
      err = np.mean(err)
      r = self.max - err
      return r

class MAE(Reward):
   """Mean Absolute Error reward function"""
   def __init__(self, norm=None, min_r=-3, max_r=1, dtype=np.float32):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r, dtype=dtype)
   def reward(self, y, ref):
      err = np.abs(y-ref)
      err = err if self.norm is None else err/self.norm
      r = self.max - np.mean( err )
      return r
      # return self.min if np.any(np.isnan(r)) else r
