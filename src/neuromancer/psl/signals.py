"""
Random signals to simulate arbitrary sequence of control actions or disturbances.
"""
import functools
import numpy as np
from numpy import dtype
from scipy import interpolate
from scipy import signal as sig
import matplotlib.pyplot as plt
import torch
from typing import Union
import warnings

_float_or_npd = Union[float, np.ndarray]
EPS = 1e-6

def _zero_one(signal):
    min, max = signal.min(axis=0, keepdims=True), signal.max(axis=0, keepdims=True)
    return (signal - min)/(max - min)


def _vectorize_and_check(arrays, d):
    for i in range(len(arrays)):
        v = arrays[i]
        arrays[i] = v.reshape(d) if isinstance(v, np.ndarray) else np.full((d,), v, dtype=np.float64)
        assert len(arrays[i]) == d and len(arrays[i].shape) == 1, "Should be a float or 1d array of length d"
    return arrays


def walk(nsim, d, min=0., max=1., sigma=0.05, bound=True, rng=np.random.default_rng()):
    """
    Gaussian random walk for arbitrary number of dimensions scaled between min/max bounds

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param sigma: (float or 1-d array) Variance of normal distribution
    :param bound: (bool) Whether to scale signal to between min and max
    :return: (np.array shape=(nsim, d)) Random walk time series of dimension d and length nsim
    """
    min, max, sigma = _vectorize_and_check([min, max, sigma], d)
    origin = np.full((1, d), 0.)
    steps = rng.normal(scale=sigma, size=(nsim-1, d))
    signal = np.concatenate([origin, steps], axis=0).cumsum(axis=0)
    return min.reshape(1, -1) + (max - min).reshape(1, -1) * _zero_one(signal) if bound else signal


def noise(nsim, d, min=0., max=1., sigma=0.05, bound=True, rng=np.random.default_rng()):
    """
    Independent Gaussian noise for arbitrary number of dimensions scaled between min/max bounds

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param sigma: (float or 1-d array) Variance of normal distribution
    :param bound: (bool) Whether to scale signal to between min and max
    :return: (np.array shape=(nsim, d)) White noise time series of dimension d and length nsim
    """
    min, max, sigma = _vectorize_and_check([min, max, sigma], d)
    signal = rng.normal(scale=sigma, size=(nsim, d))
    return min.reshape(1, -1) + (max - min).reshape(1, -1) * _zero_one(signal) if bound else signal


def step(nsim, d, min=0., max=1., randsteps=30, values=None, rng=np.random.default_rng()):
    """
    Random step function for arbitrary number of dimensions

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param randsteps: (int) Number of random steps in time series (will infer from values if values is not None)
    :param values: (np.array) An ordered list of values for each step change
    :return: (np.array shape=(nsim, d)) Time series of random steps
    """
    min, max = _vectorize_and_check([min, max], d)
    values = rng.uniform(low=min, high=max, size=(randsteps, d)) if values is None else values
    step_length = int(np.ceil(nsim / len(values)))
    signal = np.repeat(values, step_length, axis=0)[:nsim]
    return signal


_periodic_functions = {'sin': np.sin, 'square': sig.square, 'sawtooth': sig.sawtooth}


def periodic(nsim, d, min=0., max=1., periods=30, form='sin', phase_offset=False, rng=np.random.default_rng()):
    """
    periodic signals, sine, cosine

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param periods: (int) Number of periods taking place across nsim
    :return: (np.array shape=(nsim, d)) Periodic time-series
    """
    min, max = _vectorize_and_check([min, max], d)
    f = _periodic_functions[form]
    t = np.linspace(0, 2.*np.pi, nsim).reshape(-1, 1)
    offset = rng.uniform(low=0., high=2. * np.pi, size=(1, d)) if phase_offset else np.zeros((1, d))
    t = t + offset
    signal = f(periods * t)
    return min + (max - min) * _zero_one(signal)

def np_softmax(x, dim):
    return torch.nn.functional.softmax(torch.tensor(x), dim=dim).numpy()

def sines(nsim, d, min=0., max=1., periods=30, nwaves=20, form='sin', rng=np.random.default_rng()):
    """
    sum of sines

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param periods: (int) Number of periods taking place across nsim
    :return: (np.array shape=(nsim, d)) Periodic time-series
    """
    min, max = _vectorize_and_check([min, max], d)
    if periods<1:
       signal = np.ones((nsim,d))*rng.uniform(min,max)
       return signal

    f = _periodic_functions[form]
    t = np.linspace(0, 2. * np.pi, nsim).reshape(-1, 1) + rng.uniform(low=0., high=2.*np.pi, size=(1, d))
    amps = np_softmax(rng.standard_normal((nwaves, d)), dim=0)/2.
    signal = 0.5 + amps[0] * f(periods * t)
    for i in range(1, nwaves):
        periods /= 2.
        signal += 0.5 + amps[i] * f(periods * t)
    return min + (max - min) * _zero_one(signal)


def spline(nsim, d, min=0., max=1., values=None, n_interpolants=30, rng=np.random.default_rng()):
    """
    Generates a smooth cubic spline trajectory by interpolating between data points

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param n_interpolants: (int) Number of values to interpolate between
    :return: (np.array shape=(nsim, d)) Periodic time-series
    """
    min, max = _vectorize_and_check([min, max], d)
    mean = (min + max)/2.
    values = rng.triangular(min, mean, max, size=(n_interpolants, d)) if values is None else values
    cs = interpolate.CubicSpline(np.linspace(0, nsim, len(values)), values, extrapolate='periodic')
    time = np.linspace(0, nsim, nsim)
    return np.clip(cs(time), min, max)


def arma(nsim, d, min=0., max=1.,  q=10, p=10, bound=True, rng=np.random.default_rng()):
    """
    Random autoregressive moving average signal.

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :return: (np.array shape=(nsim, d)) ARMA time series
    """
    min, max = _vectorize_and_check([min, max], d)
    e = rng.normal(size=(nsim+np.max([q, p]), d))
    d = np_softmax(rng.uniform(size=(1, q)), dim=-1)
    MA = np.concatenate([d @ e[n:n+q] for n in range(nsim)], axis=0)
    c = np_softmax(rng.uniform(size=(1, p)), dim=-1)
    AR = e[:p]
    for i in range(nsim):
        AR = np.concatenate([AR, c @ AR[i:i+p]])
    ARMA = AR[p:nsim+p] + MA
    ARMA = min + (max - min) * _zero_one(ARMA) if bound else ARMA
    return ARMA


def prbs(nsim, d, min=0., max=1., p=.9, rng=np.random.default_rng()):
    """
    pseudo-random binary signal taking values min or max.

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param p: (float) probability of switching from min to max or vice versa.
    :return: (np.array shape=(nsim, d)) PRBS time series
    """
    min, max = _vectorize_and_check([min, max], d)
    signal = min.reshape(1, d)
    other = max.reshape(d)
    for i in range(nsim-1):
        switch = rng.uniform(size=d) > p
        s = [other[dim] if switch[dim] else signal[i, dim] for dim in range(d)]
        other = [signal[i, dim] if switch[dim] else other[dim] for dim in range(d)]
        signal = np.concatenate([signal, np.array(s).reshape(1, d)], axis=0)
    return signal[:nsim]

def beta(nsim, d, min=0., max=1., alpha: float=2, beta: float=2, rng=np.random.default_rng()):
    ''' 
    roughly upside down parabola shaped probability with a=b=2, support (0,1).
    The beta distribution is flexible with parameters a,b. It always has support (0,1).

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param alpha: (float) parameter of beta distribution
    :param beta: (float) parameter of beta distribution
    :param rng: (np.random.Generator) random number generator
    :return: (np.array shape=(nsim, d)) scatter of beta distributed points
    '''
    umin, umax = min, max
    b = rng.beta(a=alpha, b=beta, size=(nsim,d))
    return umin+(umax-umin)*b # transform to bounds

def _ab(m,ab):
    '''
    ab is alpha or beta depending on if m<.5,
    to achieve a frown-like shape for the pdf
    the other is set so the pdf of the beta
    dist is m.
    '''
    if m<.5:
        a=ab
        b = a*(1-m)/m
    else:
        b=ab
        a=b*m/(1-m)
    return a,b

def _beta_step_mean(x0:float, lb:float, ub:float, q:float, ab:float=2,
                    rng=np.random.default_rng()):
    """
    Generate a step from a beta distribution with mean x0
    alpha is set so that the mean is x0 given beta or vice-versa
        a/(a+b) = m
        a = m(a+b)
        a = ma+mb
        a-ma = mb
        (a-ma)/m = b
        a(1-m)/m = b
        a = bm/(1-m)
    """
    x = float(x0)
    if rng.random()<q:
        w = (ub-lb)
        x = (x-lb)/w # get x in 0-1 range
        x = np.clip(x,.1,.9) # don't get stuck on edges
        a,b = _ab(x,ab)
        x = lb + w*rng.beta(a=a,b=b)
    return x

def beta_walk_mean(nsim, d, min:_float_or_npd=0., max:_float_or_npd=1., max_step:_float_or_npd=None, x0:_float_or_npd=None,
              dtype: dtype=np.float32, ab:float=8, p:float=.5, q:float=.5, \
              rng=np.random.default_rng())->np.ndarray:
    """
    Generate a random walk from a beta distribution with parameters alpha beta set so that
    the mean is the current position and the pdf is frown-like when the other parameter is 2. 

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param max_step: (float or 1-d array) Maximum step size
    :param x0: (float or 1-d array) Initial value
    :param dtype: (dtype) Data type of output
    :param ab: (float) alpha or beta parameter of beta distribution
    :param p: (float) probability of any step
    :param q: (float) probability of individual beta step
    :param rng: (np.random.Generator) Random number generator
    :return: (np.array shape=(nsim, d)) Random walk
    """
    umin, umax = min, max
    # make sure everything is an ndarray of dtype
    umin=np.full(d,umin, dtype=dtype)
    umax=np.full(d,umax, dtype=dtype)
    x0 = (umax-umin) * rng.random(size=(d,)) + umin if x0 is None else \
        x0
    x0=np.full(d,x0, dtype=dtype)
    max_step = (umax-umin)/10 if max_step is None else \
        max_step
    max_step=np.full(d,max_step, dtype=dtype)
    np.full(d,x0, dtype=dtype)
    X = np.empty((nsim,d), dtype=dtype)
    X[0,:] = x0
    for t in range(1,nsim):
        if rng.random()<p:
            for zone in range(d):
                X[t,zone] = _beta_step_mean(x0=X[t-1,zone], lb=umin[zone], ub=umax[zone], q=q, ab=ab, rng=rng)
        else:
            X[t,:] = X[t-1,:]

    return X

def _beta_max_step(x0, lb, ub, max_step, alpha, beta, rng)->np.ndarray:
    """
    Generate a step from a beta distribution with parameters (alpha, beta)
    between lb and ub, with a maximum step size of max_step
    -> x
        lb < x < ub
    """
    x = np.array(x0).ravel()
    lb = np.maximum(lb, x-max_step)
    ub = np.minimum(ub, x+max_step)
    w = (ub-lb)
    x = lb + w*rng.beta(a=alpha,b=beta,size=x.shape).ravel()
    return x

_float_or_npd = Union[float, np.ndarray]

def beta_walk_max_step(nsim, d, min:_float_or_npd=0., max:_float_or_npd=1., max_step:_float_or_npd=None, x0:_float_or_npd=None,
              dtype: dtype=np.float32, alpha:float=2, beta:float=2, p:float=.3, \
              rng=np.random.default_rng())->np.ndarray:
    """
    Generate a random walk from a beta distribution with parameters (2,2).
    between min and max, with a maximum step size of max_step.
    No truncation is necessary, beta distro is bounded.

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param max_step: (float or 1-d array) Maximum step size
    :param x0: (float or 1-d array) Initial value
    :param dtype: (dtype) Data type of output
    :param alpha: (float) alpha parameter of beta distribution
    :param beta: (float) beta parameter of beta distribution
    :param p: (float) probability of stepping
    :param rng: (np.random.Generator) Random number generator
    :return: (np.ndarray) Random walk
    """
    umin, umax = min, max
    # make sure everything is an ndarray of dtype
    umin=np.full(d,umin, dtype=dtype)
    umax=np.full(d,umax, dtype=dtype)
    x0 = (umax-umin) * rng.random(size=(d,)) + umin if x0 is None else \
        x0
    x0=np.full(d,x0, dtype=dtype)
    max_step = (umax-umin)/10 if max_step is None else \
        max_step
    max_step=np.full(d,max_step, dtype=dtype)
    np.full(d,x0, dtype=dtype)
    X = np.empty((nsim,d), dtype=dtype)
    X[0,:] = x0
    for t in range(1,nsim):
        if rng.random()<p:
            X[t,:] = _beta_max_step(x0=X[t-1,:], lb=umin, ub=umax, max_step=max_step, alpha=alpha, beta=beta, rng=rng)
        else:
            X[t,:] = X[t-1,:]
    return X


def _eq(x,y):
    return np.abs(x-y)<10**(-4)

def _1d_walk(nsim, x0, lb, ub, max_step, p, avoid_edge, granularity, dtype, rng)->np.ndarray:
    x = x0
    ts = np.empty((nsim, 1), dtype=dtype)
    def step_len():
        if granularity==0:
            return max_step*rng.random()
        elif granularity==1:
            return max_step
        else:
            r = rng.integers(1,granularity+1)
            return max_step*r/granularity
    for i in range(nsim):
        ts[i] = x
        if rng.random()<p: # take a step with probability p
            # check if near edge and reduce chance of getting stuck there
            if _eq(x,lb):
                x = x + step_len()
            elif _eq(x,ub):
                x = x - step_len()
            else: # not stuck at edge
                if avoid_edge:
                    q = 1 - (x-lb)/(ub-lb) # prob of stepping right
                else:
                    q = .5

                if rng.random()<q: # step right
                    x = np.min((x+step_len(),ub))
                else: # step left
                    x = np.max((x-step_len(),lb))
    return ts


def nd_walk(nsim, d, min:_float_or_npd=0., max:_float_or_npd=1., x0:_float_or_npd=None, \
              max_step:_float_or_npd=None, p: float=.3, avoid_edge: bool=True, granularity: int=0, \
              dtype: dtype=np.float32, rng=np.random.default_rng())->np.ndarray:
    '''
    random walk which avoids the edge by default. If granularity is 0 takes
    uniform random steps, if granularity is 1 always takes max_step.

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param x0: (float or 1-d array) Initial value
    :param max_step: (float or 1-d array) Maximum step size
    :param p: (float) probability of stepping
    :param avoid_edge: (bool) whether to avoid the edge
    :param granularity: (int) granularity of steps (0 is uniform random, 1 is max_step, 
        >1 takes a step of length n*max_step/granularity where n is random int 1<=n<=granularity.
    :param dtype: (dtype) Data type of output
    :param rng: (np.random.Generator) Random number generator
    :return: (np.ndarray) Random walk
    '''
    umin, umax = min, max
    # make sure everything is an ndarray of dtype
    umin=np.full(d,umin, dtype=dtype)
    umax=np.full(d,umax, dtype=dtype)
    if x0 is None:
        if avoid_edge:
            x0 = (umax-umin) * rng.random(size=(d,)) + umin
        else:
            x0 = umin + (umax-umin)*rng.beta(a=alpha,b=beta,size=(d,))
    x0=np.full(d,x0, dtype=dtype)
    max_step = (umax-umin)/10 if max_step is None else \
        max_step
    max_step=np.full(d,max_step, dtype=dtype)
    X = np.empty((nsim, d), dtype=dtype)
    for zone in range(d):
        v = _1d_walk( \
                nsim=nsim, x0=x0[zone], lb=umin[zone], ub=umax[zone], max_step=max_step[zone], p=p,\
                 avoid_edge=avoid_edge, granularity=granularity, dtype=dtype, rng=rng
            ).ravel()
        X[:,zone] = v
    return X

    return None


periodic_signals = {f'{stype}': functools.partial(periodic, form=stype) for stype in _periodic_functions}
signals = {**periodic_signals, 'walk': walk, 'noise': noise, 'step': step, 'spline': spline,
           'sines': sines,  'arma': arma, 'prbs': prbs, 'beta':beta, 'beta_walk_mean':beta_walk_mean, 'beta_walk_max_step':beta_walk_max_step}


if __name__ == '__main__':
    for n, s in signals.items():
        x = s(1000, 3, min=-2., max=1.5)
        print(n, x.shape)
        plt.plot(x)
        plt.savefig(f'{n}.png')
        plt.close()