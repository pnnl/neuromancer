"""
Random signals to simulate arbitrary sequence of control actions or disturbances.
"""
import functools
import numpy as np
from scipy import interpolate
from scipy import signal as sig
import matplotlib.pyplot as plt
import torch


def _zero_one(signal):
    min, max = signal.min(axis=0, keepdims=True), signal.max(axis=0, keepdims=True)
    return (signal - min)/(max - min)


def _vectorize_and_check(arrays, d):
    for i in range(len(arrays)):
        v = arrays[i]
        arrays[i] = v.reshape(d) if isinstance(v, np.ndarray) else np.full((d,), v, dtype=np.float64)
        assert len(arrays[i]) == d and len(arrays[i].shape) == 1, "Should be a float or 1d array of length d"
    return arrays


def walk(nsim, d, min=0., max=1., sigma=0.05, bound=True):
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
    steps = np.random.normal(scale=sigma, size=(nsim-1, d))
    signal = np.concatenate([origin, steps], axis=0).cumsum(axis=0)
    return min.reshape(1, -1) + (max - min).reshape(1, -1) * _zero_one(signal) if bound else signal


def noise(nsim, d, min=0., max=1., sigma=0.05, bound=True):
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
    signal = np.random.normal(scale=sigma, size=(nsim, d))
    return min.reshape(1, -1) + (max - min).reshape(1, -1) * _zero_one(signal) if bound else signal


def step(nsim, d, min=0., max=1., randsteps=30, values=None):
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
    values = np.random.uniform(low=min, high=max, size=(randsteps, d)) if values is None else values
    step_length = int(np.ceil(nsim / len(values)))
    signal = np.repeat(values, step_length, axis=0)[:nsim]
    return signal


_periodic_functions = {'sin': np.sin, 'square': sig.square, 'sawtooth': sig.sawtooth}


def periodic(nsim, d, min=0., max=1., periods=30, form='sin', phase_offset=False):
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
    offset = np.random.uniform(low=0., high=2. * np.pi, size=(1, d)) if phase_offset else np.zeros((1, d))
    t = t + offset
    signal = f(periods * t)
    return min + (max - min) * _zero_one(signal)


def sines(nsim, d, min=0., max=1., periods=30, nwaves=20, form='sin'):
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
    f = _periodic_functions[form]
    t = np.linspace(0, 2. * np.pi, nsim).reshape(-1, 1) + np.random.uniform(low=0., high=2.*np.pi, size=(1, d))
    amps = (torch.nn.functional.softmax(torch.randn(nwaves, d), dim=0)/2.).numpy()
    signal = 0.5 + amps[0] * f(periods * t)
    for i in range(1, nwaves):
        periods /= 2.
        signal += 0.5 + amps[i] * f(periods * t)
    return min + (max - min) * _zero_one(signal)


def spline(nsim, d, min=0., max=1., values=None, n_interpolants=30):
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
    values = np.random.triangular(min, mean, max, size=(n_interpolants, d)) if values is None else values
    cs = interpolate.CubicSpline(np.linspace(0, nsim, len(values)), values, extrapolate='periodic')
    time = np.linspace(0, nsim, nsim)
    return cs(time)


def arma(nsim, d, min=0., max=1.,  q=10, p=10, bound=True):
    """
    Random autoregressive moving average signal.

    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :return: (np.array shape=(nsim, d)) ARMA time series
    """
    min, max = _vectorize_and_check([min, max], d)
    e = np.random.normal(size=(nsim+np.max([q, p]), d))
    d = torch.nn.functional.softmax(torch.tensor(np.random.uniform(size=(1, q))), dim=-1).numpy()
    MA = np.concatenate([d @ e[n:n+q] for n in range(nsim)], axis=0)
    c = torch.nn.functional.softmax(torch.tensor(np.random.uniform(size=(1, p))), dim=-1).numpy()
    AR = e[:p]
    for i in range(nsim):
        AR = np.concatenate([AR, c @ AR[i:i+p]])
    ARMA = AR[p:nsim+p] + MA
    ARMA = min + (max - min) * _zero_one(ARMA) if bound else ARMA
    return ARMA


def prbs(nsim, d, min=0., max=1., p=.9):
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
        switch = np.random.uniform(size=d) > p
        s = [other[dim] if switch[dim] else signal[i, dim] for dim in range(d)]
        other = [signal[i, dim] if switch[dim] else other[dim] for dim in range(d)]
        signal = np.concatenate([signal, np.array(s).reshape(1, d)], axis=0)
    return signal[:nsim]


periodic_signals = {f'{stype}': functools.partial(periodic, form=stype) for stype in _periodic_functions}
signals = {**periodic_signals, 'walk': walk, 'noise': noise, 'step': step, 'spline': spline,
           'sines': sines,  'arma': arma, 'prbs': prbs}


if __name__ == '__main__':
    for n, s in signals.items():
        x = s(1000, 3, min=-2., max=1.5)
        print(n, x.shape)
        plt.plot(x)
        plt.savefig(f'{n}.png')
        plt.close()