'''

additional dependencies:
colorama-0.4.5
gymnasium-0.28.1

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

from colorama import Fore, Back, Style
import gymnasium
import json
import numpy as np
from numpy import dtype
import os
from typing import Union, Sequence, Any
import warnings

from neuromancer import psl
from neuromancer.psl.nonautonomous import systems
from neuromancer.psl.building_envelope import systems as building_systems
systems.update(building_systems)

class PslGym(gymnasium.Env):
   def __init__(
         self,
         psl_sys: str,
         reward_instance: 'Reward', # see bottom of this file
         reset_freq: int=1,
         do_trace: bool=False,
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

         MSE / MAE defined in this file

      :param reset_freq: (int) number of episodes (calls to self.reset()) before reference trajectory is changed

      :param do_trace: (bool) if true keep trace of self.u_trace, self.y_trace and self.rewards per-episode

      :param p: (str), flag for printing certain things

      :param d: (str), flag for debug prints


      Fields:

      :field t: (int), current timestep (per episode)
      :field y: (numpy array), observation sim['Y'] from psl_sys
      :field d: (numpy array), disturbance sim['D'] from psl_sys
      :fields (y/u)_(min/max): (numpy array) soft min/max values
      :field refs: (numpy array), refs[t] is target for y[t+1]
      :field u: (numpy array), control action like psl_sys.get_U()
      :field psl_env: (str), name of psl.nonautonomous.ODE_NonAutonomous system
      :field observation_space: (gymnasium.spaces), includes y and refs[t]
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

      """

      super(PslGym, self).__init__()

      self.psl_env = systems[psl_sys](seed=seed)
      self.is_building = psl_sys in building_systems.keys()
      self.has_d = 'D' in self.psl_env.stats
      self.nsim = self.psl_env.nsim if psl_sys in building_systems else \
         self.psl_env.nsim-1

      self.t = 0 # timestep index
      self.episodes = 0 # track how many times self.reset() has been called

      # Set most of the attributes here
      for key in ['psl_sys','reward_instance','reset_freq','do_trace','dtype','seed','p','d']:
         setattr(self, key, locals()[key])
      for key in ['y', 'D','X','x0']: # obs, current state, initial state
         setattr(self, key, None)

      # Construct bounds, reward, and obs/act spaces
      self.nu, self.ny, self.nd, self.u_min, self.u_max, self.y_min, self.y_max, self.d_min, self.d_max = \
         self.get_bounds()
      self.U = np.empty((2,self.nu), dtype=dtype) # 2, nu because psl systems require one more action than sim steps

      # allocate memory for trace values if appropriate
      self.refs =       np.empty((self.nsim, self.ny), dtype=dtype) # reference targets

      if do_trace: # extra baggage if not used, nice for visualizing and computing dev metrics
         self.y_trace = np.empty((self.nsim, self.ny), dtype=dtype)
         self.d_trace = np.empty((self.nsim, self.nd), dtype=dtype) if self.has_d else None
         self.u_trace = np.empty((self.nsim, self.nu), dtype=dtype)
         self.u_refs  = np.empty((self.nsim, self.nu), dtype=dtype)
         self.rewards = np.empty((self.nsim, 1),       dtype=dtype)

      # Construct obs/act space:

      # FIXME: concat y | d for obs
      # ref is only y
      obs_low = np.concatenate( (self.y_min, self.y_min, self.d_min)) if self.has_d else \
         np.concatenate( (self.y_min, self.y_min))
      obs_high = np.concatenate((self.y_max, self.y_max, self.d_max)) if self.has_d else \
         np.concatenate((self.y_max, self.y_max))
      obs_size = 2*self.ny + self.nd if self.has_d else \
         2*self.ny
      self.observation_space = BoxSpace(low=obs_low, \
                                        high=obs_high, \
                                        dtype=self.dtype, shape=(obs_size,)) # include reference targets in observation

      self.action_space = BoxSpace(low=self.u_min, high=self.u_max, dtype=self.dtype, shape=(self.nu,))

      # Define reward function
      reward_norm = (self.y_max-self.y_min)

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

   def get_U(self, nsim, p=.01):
      """
      returns control actions

      signals.signals.keys()
      FIXME vary signal type for U sampling or manually sample U
      sig = inspect.signature (self.psl_env.get_U)
      if 'type' in sig.parameters.keys():
      U = self.psl_env.get_U(nsim=self.psl_env.nsim,type='step')
      signals = neuromancer.signals.keys()
      """

      if self.d: print(Fore.GREEN,f'pslgym get_U, self.psl_env.get_U(nsim={nsim})',Style.RESET_ALL)
      num_options = 4
      opt = self.psl_env.rng.integers(0,num_options)
      match opt:
         case 0:
            signal = psl.signals.step
            U = self.psl_env.get_U(nsim=nsim, signal=signal, randsteps=5)
         case 2:
            signal = psl.signals.beta_walk_max_step
            U = self.psl_env.get_U(nsim=nsim, signal=signal, p=p)
         case 3:
            signal = psl.signals.nd_walk
            U = self.psl_env.get_U(nsim=nsim, signal=signal, p=p)
         case 4:
            signal = psl.signals.beta_walk_mean
            U = self.psl_env.get_U(nsim=nsim, signal=signal, p=p)
         case _:
            signal = 'default'
            U = self.psl_env.get_U(nsim=nsim)
      signal_str = (str(signal).split('function')[-1]).split('at')[0].strip()
      if self.p:
         print(Fore.RED,signal_str,Style.RESET_ALL)
      return U, signal_str

   def set_refs(self, seed=None, type='walk')->np.array:
      '''
      get random reference trajectory from psl_env.simulate
      set self.refs
      returns initial condition

      # FIXME: add support for seed, will come with rng pull request
      # FIXME: add support for type for building envelope model
      '''
      if seed is not None:
         self.psl_env.save_random_state()
         self.psl_env.set_rng(seed)

      if self.d: print(Fore.GREEN,f'pslgym set_refs, t={self.t}, episodes={self.episodes}',Style.RESET_ALL)
      if self.p: print(Fore.GREEN,'set_refs',Style.RESET_ALL)
      # print('self.nsim',self.nsim, 'self.refs.shape',self.refs.shape)
      nsim = self.psl_env.nsim if self.is_building else self.nsim
      U, signal_str = self.get_U(nsim=self.psl_env.nsim)
      self.signal = signal_str
      x0  = self.psl_env.get_x0()
      sim = self.psl_env.simulate(U=U, x0=x0)
      if self.do_trace:
         self.u_refs[:,:] = U[:,:] if self.is_building else \
            U[:-1,:]

      if seed is not None:
         self.psl_env.restore_random_state()

      self.refs[:,:] = np.array(sim['Y'], dtype=self.dtype).reshape(self.refs.shape)

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
      u_refs  = self.u_refs
      refs    =    self.refs[:self.t,:]
      rewards = self.rewards[:self.t,:]
      y_trace = self.y_trace[:self.t,:] # include initial obs
      d_trace = self.d_trace[:self.t,:] if self.has_d else None
      mae = self.mae(ys=y_trace[1:,:], refs=refs[:-1,:])
      return mae, u_trace, u_refs, y_trace, d_trace, self.refs, rewards, self.signal

   def get_bounds(self):
      """
      Get statistics from psl for constructing observation and action spaces
      """
      u_min = np.array(self.psl_env.stats['U']['min'], dtype=self.dtype).ravel()
      u_max = np.array(self.psl_env.stats['U']['max'], dtype=self.dtype).ravel()
      y_min = np.array(self.psl_env.stats['Y']['min'], dtype=self.dtype).ravel()
      y_max = np.array(self.psl_env.stats['Y']['max'], dtype=self.dtype).ravel()
      d_min = np.array(self.psl_env.stats['D']['min'], dtype=self.dtype).ravel() if self.has_d else None
      d_max = np.array(self.psl_env.stats['D']['max'], dtype=self.dtype).ravel() if self.has_d else None

      nu = int(u_min.ravel().shape[0])
      ny = int(y_min.ravel().shape[0])
      nd = int(d_min.ravel().shape[0]) if self.has_d else None

      return nu, ny, nd, u_min, u_max, y_min, y_max, d_min, d_max

   def set_y(self, sim):
      """
      Set current observation self.obs from simulation
      also check if min/max changed
      """

      self.X = sim['X'][-1] # store psl state as initial condition
      self.y = np.array(sim['Y'][-1], dtype=self.dtype).ravel()
      self.D = np.array(sim['D'][-1], dtype=self.dtype).ravel() if self.has_d else None

      if self.do_trace:
         self.y_trace[self.t,:] = self.y

      return self.y

   def set_U(self, u):
      """
      Similar to self.set_y(), but for control actions
      """
      u = np.array(u, dtype=self.dtype).reshape(self.nu,)
      self.U[:,:] = u
      if self.do_trace:
         self.u_trace[self.t,:] = u
      
      return self.U

   def get_reward(self):
      reward = self.reward_instance.reward(y=self.y, ref=self.refs[self.t-1])
      if self.do_trace:
         self.rewards[self.t-1,:] = reward
      return reward

   def get_obs(self):
      ref = self.refs[self.t]
      return np.concatenate( (self.y, ref, self.D) ) if self.has_d else \
         np.concatenate( (self.y, ref) )

   def reset(self, seed: int=None):
      """
      Start a new episode (must be called once before step())
      """
      self.t = 0

      if self.episodes % self.reset_freq==0:
         self.x0 = self.set_refs(seed=seed)

      self.episodes = self.episodes + 1

      if self.d: print(Fore.GREEN,f'pslgym reset, t={self.t}, episodes={self.episodes}',Style.RESET_ALL)
      if self.p: print(Fore.GREEN,'reset',Style.RESET_ALL)
      U, _ = self.get_U(nsim=2)
      sim = self.psl_env.simulate(U=U,x0=self.x0,nsim=1) # get initial obs
      self.set_y(sim)

      info = {'current_time':self.t, 'num_resets':self.episodes}
      return self.get_obs(), info

   def step(self, action):
      '''
      Required Gymnasium Env method, run a single step of the simulation
      returns the observation (tuple), reward (float), terminated (bool), info (dict ['current time':self.t, 'action violation':None, 'got_NaN':False])
      '''
      self.set_U(u=action)

      truncated = False
      try:
         sim = self.psl_env.simulate(U=self.U, x0=self.X, nsim=1)
         self.t = self.t + 1
      except Exception as e:
         warnings.warn('simulaiton failed')
         print(Fore.RED,e,Style.RESET_ALL)
         truncated = True
         info = { 'time':self.t, 'episodes':self.episodes }
         return None, None, None, truncated, info

      self.set_y(sim)
      reward = self.get_reward()

      terminated = (self.t==self.nsim-1)

      obs = self.get_obs()

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
