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
import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter
from neuromancer import psl
from neuromancer.psl.nonautonomous import systems
from neuromancer.psl.building_envelope import systems as building_systems
systems.update(building_systems)
import numpy as np
from numpy import dtype
import os
import pygame
from pygame.locals import KEYDOWN, KEYUP, K_ESCAPE, K_SPACE
from typing import Union, Sequence, Any
import warnings

EPS = 1e-6

class PslGym(gymnasium.Env):
   def __init__(
         self,
         psl_sys: str,
         reward_instance: 'Reward', # see bottom of this file
         ref_fcst_l: int=3,
         reset_freq: int=1,
         do_trace: bool=False,
         dtype: dtype=np.float32,
         seed: Union[int,'RNG()']=None,
         render_mode: str=None,
         act_momentum: float=.5,
         all_D: bool=False,
         max_min_r: float=1,
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

      :param ref_fcst_l: (int) number of reference forecasts in observation

      :param reset_freq: (int) number of episodes (calls to self.reset()) before reference trajectory is changed

      :param do_trace: (bool) if true keep trace of self.u_trace, self.y_trace and self.rewards per-episode

      :param render_mode: (str) if None doesn't render, otherwise renders a pygame display every call to reset

      :param act_momentum: (float) momentum weight on previous action (reduces delta U)

      :all_D: (bool) whether to use Dhidden (as opposed to just Dout)

      :max_min_r: (float) for building systems we lower the max toward the min more so the higher max_min_r is.

      :param p: (str), flag for printing certain things

      :param d: (str), flag for debug prints

      """

      super(PslGym, self).__init__()
      self.psl_env = systems[psl_sys](seed=seed)
      self.has_d = 'D' in self.psl_env.stats
      self.is_building = psl_sys in building_systems

      self.nsim = self.psl_env.nsim

      self.t = 0 # timestep index
      self.episodes = 0 # track how many times self.reset() has been called

      # Set most of the attributes here
      for key in ['psl_sys','reward_instance','ref_fcst_l','reset_freq','do_trace','dtype','seed','render_mode','act_momentum','all_D','p','d']:
         setattr(self, key, locals()[key])
      for key in ['y','D','X','x0']:
         setattr(self, key, None)

      # Construct bounds, reward, and obs/act spaces
      self.nu, self.ny, self.nd, self.nd_hidden, self.u_min, self.umax, self.u_max, self.y_min, self.y_max, self.d_min, self.d_max, self.x_min, self.x_max = \
         self.get_bounds(max_min_r)

      self.U = np.empty((2,self.nu), dtype=dtype) # 2, nu because psl systems require one more action than sim steps
      self.a = np.empty((self.nu,), dtype=dtype)
      self.a_prev = np.empty((self.nu,), dtype=dtype)
      if self.has_d:
         self.D = np.empty((self.nsim, self.nd_hidden), dtype=dtype)

      # allocate memory for trace values if appropriate
      self.refs =       np.empty((self.nsim, self.ny), dtype=dtype) # reference targets

      if do_trace: # extra baggage if not used, nice for visualizing and computing dev metrics
         self.y_trace = np.empty((self.nsim, self.ny), dtype=dtype)
         self.u_trace = np.empty((self.nsim-1, self.nu), dtype=dtype)
         self.u_refs  = np.empty((self.nsim-1, self.nu), dtype=dtype)
         self.rewards = np.empty((self.nsim-1, 1),       dtype=dtype)
         self.episode_rewards = []

      # Construct observation and  action spaces:
      # action space is always normalized to range -1,1
      self.action_space = BoxSpace(low=np.full(self.nu, -1, dtype=self.dtype),
                                   high=np.full(self.nu, 1, dtype=self.dtype),
                                   dtype=self.dtype, shape=(self.nu,))

      # 1 (current obs) + ref_fcst_l (number of reference forecasts)
      r_ref_min = [self.y_min for n in range(1+self.ref_fcst_l)]
      # Include reference targets in observaiton
      self.obs_low = np.concatenate( r_ref_min + [self.d_min] ) if self.has_d else \
         np.concatenate( r_ref_min )

      r_ref_max = [self.y_max for n in range(1+self.ref_fcst_l)]
      self.obs_high = np.concatenate( r_ref_max + [self.d_max] ) if self.has_d else \
         np.concatenate( r_ref_max )

      # Check where low is too close to high and shift them apart by scaled epsilon
      low_eq_high = np.where( np.abs( self.obs_high-self.obs_low ) < EPS*( np.maximum(1, np.abs(self.obs_high)) ) )
      self.obs_low[low_eq_high] = self.obs_low[low_eq_high] - EPS*( np.maximum(1, np.abs(self.obs_low[low_eq_high])) )
      self.obs_high[low_eq_high] = self.obs_high[low_eq_high] + EPS*( np.maximum(1, np.abs(self.obs_high[low_eq_high])) )

      obs_low = np.concatenate((self.action_space.low, np.full(self.obs_low.shape[0], -np.inf, dtype=self.dtype)))
      obs_high = np.concatenate((self.action_space.high, np.full(self.obs_low.shape[0], np.inf, dtype=self.dtype)))
      obs_size = self.action_space.shape[0] + self.ny*(1+self.ref_fcst_l) + self.nd
      self.observation_space = BoxSpace(low=obs_low,
                                        high=obs_high,
                                        dtype=self.dtype, shape=(obs_size,))
      # Define reward function
      reward_norm = (self.y_max-self.y_min)

      # instantiate Reward if class was passed
      if not isinstance(self.reward_instance, Reward):
         self.reward_instance = self.reward_instance(norm=reward_norm)
      else:
         self.reward_instance.set_norm(reward_norm)

      self.plt = PlotEnv(env=self, render_mode=self.render_mode)

   def get_U(self, nsim, p=.01):
      """
      returns control actions randomly generated from signals
      """

      num_options = 4
      opt = self.psl_env.rng.integers(0,num_options)
      match opt:
         case 0:
            signal = psl.signals.step
            U = self.psl_env.get_U(nsim=nsim, umin=self.u_min, umax=self.umax, signal=signal, randsteps=5)
         case 1:
            signal = psl.signals.beta_walk_max_step
            U = self.psl_env.get_U(nsim=nsim, umin=self.u_min, umax=self.umax, signal=signal, p=p)
         case 2:
            signal = psl.signals.nd_walk
            U = self.psl_env.get_U(nsim=nsim, umin=self.u_min, umax=self.umax, signal=signal, p=p)
         case 3:
            signal = psl.signals.beta_walk_mean
            U = self.psl_env.get_U(nsim=nsim, umin=self.u_min, umax=self.umax, signal=signal, p=p)

      signal_str = (str(signal).split('function')[-1]).split('at')[0].strip()
      return U, signal_str

   def set_refs(self, seed=None):
      '''
      get random reference trajectory from psl_env.simulate or directly generate references
      set self.refs
      returns initial condition
      '''
      if seed is not None:
         self.psl_env.save_random_state()
         self.psl_env.rng = np.random.default_rng(seed=seed)

      U, signal_str = self.get_U(nsim=self.nsim+1)
      x0  = self.psl_env.rng.uniform(low=self.x_min, high=self.x_max)
      sim = self.psl_env.simulate(nsim=self.nsim, U=U, x0=x0)
      if self.do_trace:
         self.u_refs[:,:] = U[2:,:]

      if seed is not None:
         self.psl_env.restore_random_state()

      if self.psl_env.rng.uniform()<.5:
         self.refs[:,:] = np.array(sim['Y'], dtype=self.dtype).reshape(self.refs.shape)
         self.signal = signal_str
      else:
         if self.is_building:
            y_min=18
            y_max=26
            p=.003
            periods=self.psl_env.rng.integers(6)
         else:
            y_min=self.y_min
            y_max=self.y_max
            p=.01
            periods=self.psl_env.rng.integers(4)
         if self.psl_env.rng.uniform()<.5:
            self.refs[:,:] = psl.signals.beta_walk_mean(nsim=self.refs.shape[0], d=self.ny, min=y_min, max=y_max, p=p, rng=self.psl_env.rng)
            self.signal = 'ref ~ beta_mean'
         else:
            self.refs[:,:] = psl.signals.sines(nsim=self.refs.shape[0], d=self.ny, min=y_min, max=y_max, periods=periods, rng=self.psl_env.rng)
            self.signal = 'ref ~ sines'

         self.u_refs[:,:] = 0


      if self.has_d:
         self.D[:,:] = np.array(sim['Dhidden'], dtype=self.dtype).reshape(self.D.shape)

      return x0

   def set_do_trace(self,do_trace: bool=True):
      '''
      WARNING: must call reset() immediately after set_do_trace()
      '''
      self.do_trace = do_trace

   def mae(self, ys, refs):
      '''
      returns mean absolute error
      '''
      assert self.do_trace, 'mae only works if self.do_trace'
      if ys.shape[0]==0:
         return None
      err = ys-refs
      mae = float(np.mean( np.sum( np.abs( err ), axis=1).ravel() ) )
      return mae

   def get_trace(self, t=None):
      '''
      return trace of values for visualization and computing metrics
      not necessarily part of training process
      '''
      if t is None:
         t = self.t
      assert self.do_trace, 'get_trace only works if self.do_trace'
      y_trace = self.y_trace[:t+1,:] # include initial obs
      refs    =    self.refs[:t+1,:]
      mae = self.mae(ys=y_trace[1:,:], refs=refs[:-1,:])
      u_trace = self.u_trace[:t,:]
      rewards = self.rewards[:t,:]
      return mae, self.normalize_act(u_trace), self.normalize_act(self.u_refs), y_trace, self.D, self.refs, rewards, self.signal

   def get_bounds(self, max_min_r):
      """
      Get statistics from psl for constructing observation and action spaces
      and for (de)normalization
      """
      u_min = np.array(self.psl_env.stats['U']['min'], dtype=self.dtype).ravel() if not hasattr(self.psl_env,'umin') else \
         self.psl_env.umin
      u_max = np.array(self.psl_env.stats['U']['max'], dtype=self.dtype).ravel() if not hasattr(self.psl_env,'umax') else \
         self.psl_env.umax
      y_min = np.array(self.psl_env.stats['Y']['min'], dtype=self.dtype).ravel()
      y_max = np.array(self.psl_env.stats['Y']['max'], dtype=self.dtype).ravel()
      if self.has_d:
         if self.all_D:
            d_min = np.array(self.psl_env.stats['Dhidden']['min'], dtype=self.dtype).ravel()
            d_max = np.array(self.psl_env.stats['Dhidden']['max'], dtype=self.dtype).ravel()
            nd = d_min.ravel().shape[0]
         else:
            d_min = np.array(self.psl_env.stats['D']['min'], dtype=self.dtype).ravel()
            d_max = np.array(self.psl_env.stats['D']['max'], dtype=self.dtype).ravel()
            nd = d_min.ravel().shape[0]
   
         nd_hidden = self.psl_env.stats['Dhidden']['min'].ravel().shape[0]
      else:
         d_min, d_max, nd, nd_hidden = None, None, 0, 0

      x_max = self.psl_env.stats['X']['max']
      x_min = self.psl_env.stats['X']['min']
      if self.is_building: # reduce upper bounds for sampling actions and initial state
         umax = (u_max+max_min_r*u_min)/(1+max_min_r)
         x_max = (x_max+max_min_r*x_min)/(1+max_min_r)
      else:
         umax = u_max

      nu = int(u_min.ravel().shape[0])
      ny = int(y_min.ravel().shape[0])

      return nu, ny, nd, nd_hidden, u_min, umax, u_max, y_min, y_max, d_min, d_max, x_min, x_max

   def set_y(self, sim, y=None):
      """
      Set current observable self.y from simulation
      store current state X (only for passing back to sim)
      agent does not know X.
      """

      self.X = sim['X'][-1] # store psl state as initial condition
      self.y = np.array(sim['Y'][-1], dtype=self.dtype).ravel() if y is None else \
         y

      if self.do_trace:
         self.y_trace[self.t,:] = self.y

      return self.y

   def denorm_act(self, a):
      '''
      a: an element of self.action_space in range (-1,1)
      output u in physical units range (self.u_min, self.u_max)
      '''
      return self.u_min + (self.u_max - self.u_min) * (a+1)/2

   def normalize_act(self, u):
      '''
      opposite of denorm_act
      '''
      return 2 * (u - self.u_min) / (self.u_max - self.u_min) - 1

   def set_U(self, a):
      """
      track a_prev, average a with self.a, set self.U as denormalized
      duplicated self.a
      """
      a = np.array(a, dtype=self.dtype).reshape(self.nu,)

      self.a_prev[:] = self.a[:]
      self.a[:] = a + self.act_momentum * (self.a-a)

      u = self.denorm_act(self.a.copy())
      self.U[:,:] = u
      if self.do_trace:
         self.u_trace[self.t,:] = u
      
      return self.U

   def get_reward(self):
      '''
      set previous reward based on current y and previous ref,
      current and previous actions (delta U penalty)
      '''
      reward = self.reward_instance.reward(y=self.y, ref=self.refs[self.t-1], a=self.a, a_prev=self.a_prev)
      if self.do_trace:
         self.rewards[self.t-1,:] = reward
      return reward

   def norm_obs(self, obs):
      '''
      normalize last part of observation.
      Note self.obs_low/self.obs_high do not include action, which is technically part of the observation:
      obs_full = (a, y, refs, d)
      obs      =    (y, refs, d)
      '''
      return 2 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1

   def obs_to_y(self, obs):
      '''
      as opposed to norm_obs, this obs is the full observation.
      obs = (a, y, refs, d)
      '''
      y = obs[self.nu: self.nu+self.ny]
      y = self.y_min + (self.y_max-self.y_min) * (y+1)/2
      return y

   def get_obs(self) -> np.ndarray:
      '''
      constructs the full normalized observation from the current values
      output: array(a, y, refs, d)
      '''
      ref = np.empty((self.ref_fcst_l,self.ny), dtype=self.dtype)
      idx_f = self.t+self.ref_fcst_l
      if idx_f<=self.nsim:
         ref = self.refs[self.t:idx_f]
      else:
         rmdr = idx_f-self.nsim
         ref[:-rmdr,:] = self.refs[self.t:]
         ref[-rmdr:,:] = ref[-(rmdr+1)]
      if self.has_d:
         if self.all_D:
            D = self.D[self.t]
         else:
            D = self.D[self.t, self.psl_env.d_idx]
      else:
         D = None
      obs = np.concatenate( (self.a, self.y, ref.ravel(), D) ) if self.has_d else \
         np.concatenate( (self.a, self.y, ref.ravel()) )

      # note that a is already normalized
      obs[self.nu:] = self.norm_obs(obs[self.nu:])

      return obs

   def render(self):
      """
      Visualize y, refs, rewards, actions
      should have the option to save a plot especially at the end of an episode, and save plot in reset().
      Note this method is bypassed by explicitly rendering in reset.
      If real-time rendering is desired comment pass and uncomment the rest.
      """
      pass
      # if self.render_mode is not None:
      #    if self.t%20==0 and self.t>0:
      #       self.plt.render()

   def reset(self, seed: int=None, options=None):
      """
      Start a new episode (must be called once before step())
      renders pygame display if self.render_mode is not None
      """
      if self.episodes>0:
         if self.t==0: # handles redundant resetting from dummy_vec_env.py
            info = {'current_time':self.t, 'num_resets':self.episodes, 'TimeLimit.truncated':False}
            return self.get_obs(), info

         if self.render_mode is not None:
            self.episode_rewards.append(np.mean(self.rewards.ravel()))
            self.plt.render(self.nsim-1)

      self.t = 0

      if self.episodes % self.reset_freq == 0:
         self.x0 = self.set_refs(seed=seed)

      self.episodes = self.episodes + 1

      U, _ = self.get_U(nsim=2)
      a = self.normalize_act(U[0,:])
      self.a[:] = a[:]

      self.a_prev[:] = a[:]
      sim = self.psl_env.simulate(nsim=1, U=U, x0=self.x0,) # get initial obs
      self.set_y(sim)

      info = {'current_time':self.t, 'num_resets':self.episodes, 'TimeLimit.truncated':False}
      return self.get_obs(), info

   def step(self, action):
      '''
      Required Gymnasium Env method, run a single step of the simulation
      returns the observation (tuple), reward (float), terminated (bool), info (dict)
      '''
      self.set_U(a=action)

      if self.has_d:
         sim = self.psl_env.simulate(U=self.U, x0=self.X, nsim=1, D=self.D[self.t:self.t+1])
      else:
         sim = self.psl_env.simulate(U=self.U, x0=self.X, nsim=1)

      self.t = self.t + 1

      self.set_y(sim)

      reward = self.get_reward()

      obs = self.get_obs()

      terminated = (self.t == self.nsim-1)
      info = {'time': self.t, 'episodes': self.episodes, 'TimeLimit.truncated':False}
      truncated=False
      return obs, reward, terminated, truncated, info


def format_func(value, tick_number):
    return f'{value:.2f}'  # Format the number as a decimal with 2 places of precision

formatter = FuncFormatter(format_func)

plot_shape = (1600,800)
class PlotEnv():
   def __init__(self, env, dpi=96, render_mode=None):
      global plot_shape
      self.env = env
      self.dpi = dpi
      self.black = (0, 0, 0)
      if render_mode is not None:
         self.screen = pygame.display.set_mode(plot_shape, pygame.RESIZABLE)

   def plot(self,t):
      global plot_shape

      env = self.env
      mae, us, ur, ys, ds, refs, rs, sig = env.get_trace(t)
      colors = cm.get_cmap('viridis', us.shape[1])
      r = rs[-1,:]
      ts = np.arange(0 ,env.nsim)
      refs = refs
      axs = []
      for event in pygame.event.get():
         if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
               exit()
         elif event.type==pygame.VIDEORESIZE:
            plot_shape = (event.w, event.h)
      fig = plt.figure(figsize=(plot_shape[0]/self.dpi, plot_shape[1]/self.dpi), dpi=self.dpi)
      gg = gridspec.GridSpec(env.ny+1, 2, width_ratios=[1, 1])
      for i in range(env.ny):
         ax = plt.subplot(gg[i, 0])
         ax.plot(ys[:,i],color='blue')
         ax.plot(ts, refs[:,i], color='red',linestyle='dashed')
         ax.legend([f'y[{i}]','reference'])
         axs.append(ax)
      ax = plt.subplot(gg[env.ny, 0])
      ax.plot(rs, color='green')
      ax.yaxis.set_major_formatter(formatter)
      ax.legend(['reward'])
      ax.set_xlabel('timestep')
      axs.append(ax)

      ax = plt.subplot(gg[0, 1])
      ax.set_title('reward per episode')
      ax.scatter(range(self.env.episodes),[self.env.episode_rewards[n] for n in range(self.env.episodes)])
      ax.set_xlabel('episode')
      ax.set_ylabel('mean reward')
      axs.append(ax)

      ax = plt.subplot(gg[1:, 1])
      ax.set_title('reference signal: '+sig)
      for i in range(ur.shape[1]):
         ax.plot(ur[:,i],linestyle='dashed',color=colors(i))
         ax.plot(us[:,i],color=colors(i))
      ax.set_ylim([-1, 1])      
      axs.append(ax)
      r = np.round(float(r.mean()),3)
      mae = np.round(mae,3) if mae is not None else None
      mae = 'N/A' if mae is None else '{:.3f}'.format(mae)

      title = f'{self.env.psl_sys}: Mean Reward {r}, MAE {mae}, t={self.env.t}\n'
      plt.suptitle(r'{}'.format(title))
      plt.tight_layout()
      plt.legend(['reference actions'])
      return fig, axs
   def render(self, t):
      fig, axs = self.plot(t)
   # https://stackoverflow.com/questions/48093361/using-matplotlib-in-pygame v
      canvas = agg.FigureCanvasAgg(fig)
      canvas.draw()
      renderer = canvas.get_renderer()
      raw_data = renderer.tostring_rgb()
      size = canvas.get_width_height()
      surf = pygame.image.fromstring(raw_data, size, "RGB")
      # self.screen.fill(self.black)
      self.screen.blit(surf, (0,0))
   # https://stackoverflow.com/questions/48093361/using-matplotlib-in-pygame ^
      plt.close(fig)
      events = pygame.event.get()
      pygame.display.flip()



# Gymnasium Spaces:

class BoxSpace(gymnasium.spaces.Box):
   """
   Wrapper for Box space, more robust contains() method
   """
   def __init__(self, low, high, shape=None, dtype=np.float32):
      super().__init__(low=low, high=high, shape=shape, dtype=dtype)
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
   def reward(self, y: np.array, ref: np.array, norm: Union[float, np.array], a: Union[float, np.array]=None, a_prev: Union[float, np.array]=None)->float:
      raise NotImplementedError

class MSE(Reward):
   """MSE reward function"""
   def __init__(self, norm=None, min_r=-3, max_r=1, dtype=np.float32):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r, dtype=dtype)
   def reward(self, y, ref, a=None, a_prev=None):
      err = (y-ref)**2
      err = err if self.norm is None else err/(self.norm**2)
      err = err.ravel()
      err = np.mean(err)
      if a is not None:
         err_a = (a-a_prev)**2
         err_a = err_a.ravel()
         err_a = np.mean(err_a)
         # err = err + (err_a/8)*(err_a>.01)
         # err = err + (err_a/8)
         err = err + (err_a/16)
         # err = err + (err_a/4)

      r = self.max - err
      return r

class MAE(Reward):
   """Mean Absolute Error reward function"""
   def __init__(self, norm=None, min_r=-3, max_r=1, dtype=np.float32):
      super().__init__(norm=norm, min_r=min_r, max_r=max_r, dtype=dtype)
   def reward(self, y, ref, a=None, a_prev=None):
      err = np.abs(y-ref)
      err = err if self.norm is None else err/self.norm
      r = self.max - np.mean( err )
      return r
      # return self.min if np.any(np.isnan(r)) else r
