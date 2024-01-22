"""
Base Control Profiles for System excitation
# TODO: All signals should be nsim X nx np.arrays of type np.float64.
# TODO: Any signals bounded by xmin and xmax should be tested to ensure this.
# TODO: No camel case functions
# TODO: No random seed setting
# TODO: Only variables with upper case should be arrays which are matrices (this is discouraged but acceptable)
"""

import numpy as np
import random as rd
from scipy import interpolate
from scipy import signal as sig


def RandomWalk(nx=1, nsim=100, xmax=1, xmin=0, sigma=0.05, rseed=1):
    """

    :param nx: (int) State space dimension
    :param nsim: (int) Number of simulation steps
    :param xmax: (float) Upper bound on state values
    :param xmin: (float) Lower bound on state values
    :param sigma: (float) Variance of normal distribution
    :param rseed: (int) Set random seed
    :return:
    """

    rd.seed(rseed)

    if type(xmax) is not np.ndarray:
        xmax = np.asarray(nx*[xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray(nx*[xmin]).ravel()

    Signals = []
    for k in range(nx):
        Signal = [0]
        for t in range(1, nsim):
            yt = Signal[t - 1] + np.random.normal(0, sigma)
            if (yt > 1):
                yt = Signal[t - 1] - abs(np.random.normal(0, sigma))
            elif (yt < 0):
                yt = Signal[t - 1] + abs(np.random.normal(0, sigma))
            Signal.append(yt)
        Signals.append(xmin[k] + (xmax[k] - xmin[k])*np.asarray(Signal))
    return np.asarray(Signals).T


def random_walk(nsim, d, min=0., max=1., sigma=0.05):
    """
    Gaussian random walk for arbitrary number of dimensions scaled between min/max bounds
    TODO: Test within min and max. Test nsim, d, edge case d = 1.
    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions for the random walk
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param sigma: (float or 1-d array) Variance of normal distribution
    :return: (np.array shape=(nsim, d)) Random walk time series of dimension nx and length nsim
    """
    max = max if isinstance(max, np.ndarray) else np.full((d,), max, dtype=np.float64)
    min = min if isinstance(min, np.ndarray) else np.full((d,), min, dtype=np.float64)
    sigma = sigma if isinstance(sigma, np.ndarray) else np.full((d,), sigma, dtype=np.float64)
    assert len(sigma) == d and len(sigma.shape) == 1, "sigma should be a float or 1d array of length d"
    assert len(max) == d and len(max.shape) == 1, "max should be a float or 1d array of length d"
    assert len(min) == d and len(min.shape) == 1, "min should be a float or 1d array of length d"

    origin = np.full((1, d), 0.)
    steps = np.random.normal(scale=sigma, size=(nsim-1, d))
    signal = np.concatenate([origin, steps], axis=0).cumsum(axis=0)
    return min + (max - min)*signal


def WhiteNoise(nx=1, nsim=100, xmax=1, xmin=0, rseed=1):
    """
    White Noise
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param rsee: (int) Set random seed
    """

    # rd.seed(rseed)

    if type(xmax) is not np.ndarray:
        xmax = np.asarray(nx*[xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray(nx*[xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = xmin[k] + (xmax[k] - xmin[k])*np.random.rand(nsim)
        Signal.append(signal)
    return np.asarray(Signal).T


def white_noise(nsim, d, min=0., max=1., sigma=0.05):
    """
    Gaussian random walk for arbitrary number of dimensions scaled between min/max bounds
    :param nsim: (int) Number of simulation steps
    :param d: (int) Number of dimensions for the random walk
    :param min: (float or 1-d array) Lower bound on values
    :param max: (float or 1-d array) Upper bound on values
    :param sigma: (float or 1-d array) Variance of normal distribution
    :return: (np.array shape=(nsim, d)) Random walk time series of dimension nx and length nsim
    """
    signal = np.random.normal(scale=sigma, size=(nsim, d))
    return min + (max - min) * signal


def Step(nx=1, nsim=100, tstep=50, xmax=1, xmin=0, rseed=1):
    """
    step change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param tstep: (int) time of the step
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param rseed: (int) Set random seed
    """

    # rd.seed(rseed)

    if type(xmax) is not np.ndarray:
        xmax = np.asarray(nx * [xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray(nx * [xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = np.ones(nsim)
        signal[0:tstep] = xmin[k]
        signal[tstep:] = xmax[k]
        Signal.append(signal)
    return np.asarray(Signal).T


def Steps(nx=1, nsim=100, values=None, randsteps=5, xmax=1, xmin=0, rseed=1):
    """

    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param values: (list/ndarray) sequence of step changes, e.g., [0.4, 0.8, 1, 0.7, 0.3, 0.0]
    :param randsteps: (int) number of random step changes if values is None
    :param xmax: (int/ndarray) signal maximum value
    :param xmin: (int/ndarray) signal minimum value
    :param rseed: (int) Set random seed
    :return:
    """

    # rd.seed(rseed)

    if values is None:
        values = np.round(np.random.rand(randsteps), 3)
    if type(values) is not np.ndarray:
        values = np.asarray([values]).ravel()
    if type(xmax) is not np.ndarray:
        xmax = np.asarray(nx*[xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray(nx*[xmin]).ravel()

    step_length = int(np.ceil(nsim/len(values)))
    signal = np.ones([nx, nsim])
    for j in range(nx):
        for k in range(len(values)):
            signal[j, k*step_length:(k+1)*step_length] = values[k]*(xmax[j]-xmin[j])+xmin[j]
    return signal.T


def Sawtooth(nx=1, nsim=100, numPeriods=1, xmax=1, xmin=0, rseed=1):
    """
    ramp change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param numPeriods: (int) Number of periods
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param rseed: (int) Set random seed
    """

    rd.seed(rseed)

    assert nsim >= numPeriods, 'numPeriods must be smaller than nsim'
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax] * nx).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin] * nx).ravel()

    xmax = xmax.reshape(nx)
    xmin = xmin.reshape(nx)

    t = np.linspace(0, 1, nsim)
    Signal = []
    for k in range(nx):
        signal = xmin[k] + (xmax[k] - xmin[k])*(0.5 * (sig.sawtooth(2 * np.pi * numPeriods * t) + 1))
        Signal.append(signal)
    return np.asarray(Signal).T


def Periodic(nx=1, nsim=100, numPeriods=1, xmax=1, xmin=0, form='sin', rseed=1):
    """
    periodic signals, sine, cosine
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param numPeriods: (int) Number of periods
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param form: (str) form of the periodic signal 'sin' or 'cos'
    :param rseed: (int) Set random seed
    """

    rd.seed(rseed)

    assert nsim >= numPeriods, 'numPeriods must be smaller than nsim'
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]*nx).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]*nx).ravel()

    xmax = xmax.reshape(nx)
    xmin = xmin.reshape(nx)

    samples_period = nsim// numPeriods
    leftover = nsim % numPeriods
    Signal = []
    extraPeriods = 0
    if leftover > samples_period:
        extraPeriods = leftover//samples_period
        leftover = leftover % samples_period
    for k in range(nx):
        if form == 'sin':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        elif form == 'cos':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        elif form == 'square':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * sig.square(np.arange(0, 2*np.pi, 2*np.pi / samples_period)))
        signal = np.tile(base, numPeriods+extraPeriods)
        signal = np.append(signal, base[0:leftover])
        Signal.append(signal)
    return np.asarray(Signal).T

  
def SplineSignal(nsim=500, values=None, xmin=0, xmax=1, rseed=1):
    """
    Generates a smooth cubic spline trajectory by interpolating between data points

    :param nsim: (int) Number of simulation steps
    :param values: (np.array) values to interpolate
    :param xmin: (float) Minimum value of time series
    :param xmax: (float) Maximum value of time series
    :param rseed: (int) Set random seed.
    :return:
    """
    if values is None:
        rd.seed(rseed)
        values = [rd.triangular(xmin, xmax) for _ in range(30)]
    dt = int(np.ceil(nsim / len(values)))
    dt_time = np.arange(0, nsim, dt)
    cs = interpolate.CubicSpline(dt_time, values[:len(dt_time)], extrapolate='periodic')
    time = np.arange(0, nsim)
    return cs(time)
