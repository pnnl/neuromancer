"""
wrapper for emulator dynamical models
Internal Emulators - in house ground truth equations
External Emulators - third party models
"""

from scipy.io import loadmat
from scipy import signal
from abc import ABC, abstractmethod
import numpy as np
import plot
import matplotlib.pyplot as plt


####################################
###### Internal Emulators ##########
####################################

class EmulatorBase(ABC):
    """
    base class of the emulator
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    @abstractmethod
    def parameters(self, **kwargs):
        pass

    # equations defining the dynamical system
    @abstractmethod
    def equations(self, **kwargs):
        pass

    # N-step forward simulation of the dynamical system
    @abstractmethod
    def simulate(self, **kwargs):
        pass


"""
Linear ODEs
"""

class LTISSM(EmulatorBase):
    """
    base class of the linear time invariant state space model
    """
    def __init__(self):
        super().__init__()
        pass


class ExpGrowth(EmulatorBase):
    """
    exponentia growth linear ODE
    https://en.wikipedia.org/wiki/Exponential_growth
    """
    def __init__(self):
        super().__init__()
        pass

    def parameters(self):
        self.x0 = 1
        self.nx = 1
        self.k = 2
        self.A = self.k*np.eye(self.nx)

    # equations defining single step of the dynamical system
    def equations(self, x):
        x = self.A*x
        return x

    def simulate(self, ninit, nsim, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        X= []
        for k in range(nsim):
            x = self.equations(x)
            X.append(x)  # updated states trajectories
        return np.asarray(X)


class SimpleHarmonicMotion(EmulatorBase):
    """
    https://en.wikipedia.org/wiki/Simple_harmonic_motion
    """
    def __init__(self):
        super().__init__()
        pass



"""
Nonlinear ODEs

list of nlin ODEs
https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
"""



class LogisticGrowth(EmulatorBase):
    """
    logistic growth linear ODE
    https://en.wikipedia.org/wiki/Logistic_function
    """
    def __init__(self):
        super().__init__()
        pass

    def parameters(self):
        self.k = 2
        self.x0 = 1
        self.nx = 1

    # equations defining single step of the dynamical system
    def equations(self, x):
        pass

    def simulate(self, ninit, nsim, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        X= []
        for k in range(nsim):
            x = self.equations(x)
            X.append(x)  # updated states trajectories
        return np.asarray(X)



class Building_hf(EmulatorBase):
    """
    building heat transfer model with linear state dynamics and bilinear heat flow input dynamics
    represents building envelope with radiator for zone heating
    parameters obtained from the original white-box model from Modelica
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self, file_path='./emulators/buildings/Reno_model_for_py.mat'):
        file = loadmat(file_path)

        #  LTI SSM model
        self.A = file['Ad']
        self.B = file['Bd']
        self.C = file['Cd']
        self.D = file['Dd']
        self.E = file['Ed']
        self.G = file['Gd']
        self.F = file['Fd']
        #  constraints bounds
        self.Ts = file['Ts']  # sampling time
        self.TSup = file['TSup']  # supply temperature
        self.umax = file['umax']  # max heat per zone
        self.umin = file['umin']  # min heat per zone
        self.mf_max = self.umax / 20  # maximal nominal mass flow l/h
        self.mf_min = self.umin / 20  # minimal nominal mass flow l/h
        self.dT_max = 40  # maximal temperature difference deg C
        self.dT_min = 0  # minimal temperature difference deg C
        #         heat flow equation constants
        self.rho = 0.997  # density  of water kg/1l
        self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
        self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds
        # problem dimensions
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.nd = self.E.shape[1]
        self.n_mf = self.B.shape[1]
        self.n_dT = 1
        # initial conditions and disturbance profiles
        self.x0 = 0 * np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.D = file['disturb'] # pre-defined disturbance profiles

    # equations defining single step of the dynamical system
    def equations(self, x, m_flow, dT, d):
        u = m_flow * self.rho * self.cp * self.time_reg * dT
        x = np.matmul(self.A, x) + np.matmul(self.B, u) + np.matmul(self.E, d) + self.G.ravel()
        y = np.matmul(self.C, x) + self.F.ravel()
        return u, x, y

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, M_flow, DT, D=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param M_flow: (ndarray, shape=(nsim, self.n_mf)) mass flow profile matrix
        :param DT: (ndarray, shape=(nsim, self.n_dT)) temperature difference profile matrix
        :param D: (ndarray, shape=(nsim, self.nd)) measured disturbance signals
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response matrices, i.e. U, X, Y, for heat flows, states, and output ndarrays
        """
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0

        if D is None:
            D = self.D[ninit: ninit+nsim,:]

        U, X, Y = [], [], []
        N = 0
        for m_flow, dT, d in zip(M_flow, DT, D):
            N += 1
            u, x, y = self.equations(x, m_flow, dT, d)
            U.append(u)
            X.append(x + 20)  # updated states trajectories with initial condition 20 deg C of linearization
            Y.append(y - 273.15)  # updated input trajectories from K to deg C
            if N == nsim:
                break
        return np.asarray(U), np.asarray(X), np.asarray(Y)


class Building_hf_ROM(Building_hf):
    """
    Reduced order building heat transfer model with linear
    state dynamics and bilinear heat flow input dynamics
    represents building envelope with radiator for zone heating
    parameters obtained from the original white-box model from Modelica
    discrete time state space model form:
    x_{k+1} = A x_k + B u_k + E d_k
    u_k = a_k H b_k
    y_k = C x_k
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self, file_path='./emulators/buildings/Reno_model_for_py.mat'):
        super().parameters(file_path)
        file = loadmat(file_path)
        #  LTI SSM model
        self.A = file['Ad_ROM']
        self.B = file['Bd_ROM']
        self.C = file['Cd_ROM']
        self.D = file['Dd_ROM']
        self.E = file['Ed_ROM']
        self.G = file['Gd_ROM']
        self.F = file['Fd_ROM']

        #  constraints bounds
        self.Ts = file['Ts']  # sampling time
        self.TSup = file['TSup']  # supply temperature
        self.umax = file['umax']  # max heat per zone
        self.umin = file['umin']  # min heat per zone
        self.mf_max = self.umax / 20  # maximal nominal mass flow l/h
        self.mf_min = self.umin / 20  # minimal nominal mass flow l/h
        self.dT_max = 40  # maximal temperature difference deg C
        self.dT_min = 0  # minimal temperature difference deg C
        #         heat flow equation constants
        self.rho = 0.997  # density  of water kg/1l
        self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
        self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds
        # problem dimensions
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.nd = self.E.shape[1]
        self.n_mf = self.B.shape[1]
        self.n_dT = 1
        # initial conditions and disturbance profiles
        self.x0 = 0 * np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.D = file['disturb']  # pre-defined disturbance profiles


"""
Linear PDEs
"""



"""
Nonlinear PDEs

list of nlin PDEs
https://en.wikipedia.org/wiki/List_of_nonlinear_partial_differential_equations
"""



"""
Chaotic systems and fractals

chaotic systems
https://en.wikipedia.org/wiki/List_of_chaotic_maps
"""


class LorentzAttractor(EmulatorBase):
    """
    base class of the linear time invariant state space model
    TODO: apply
    https://en.wikipedia.org/wiki/Lorenz_system#Analysis
    # https://ipywidgets.readthedocs.io/en/stable/examples/Lorenz%20Differential%20Equations.html
    # https://scipython.com/blog/the-lorenz-attractor/
    # https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    """
    def __init__(self):
        super().__init__()
        pass

class VanDerPolOsscilator(EmulatorBase):
    """
    base class of the linear time invariant state space model
    TODO: apply
    # http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    http://hadron.physics.fsu.edu/~eugenio/comphy/examples/vanderpol.py
    https://www.johndcook.com/blog/2019/12/26/driven-van-der-pol/
    https://www.johndcook.com/blog/2019/12/22/van-der-pol/
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    """
    def __init__(self):
        super().__init__()
        pass


class Mandelbrot(EmulatorBase):
    """
    base class of the linear time invariant state space model
    TODO: apply
    https://www.geeksforgeeks.org/mandelbrot-fractal-set-visualization-in-python/
    #https://scipy-lectures.org/intro/numpy/auto_examples/plot_mandelbrot.html
    https://rosettacode.org/wiki/Mandelbrot_set
    https://levelup.gitconnected.com/mandelbrot-set-with-python-983e9fc47f56
    IDEA: use mandelbrot zoom video as our dataset for training
    Cool effect
    """
    def __init__(self):
        super().__init__()
        pass



"""
TODO: bunch of open source physics implementations to be integrated in the framework
# http://www-personal.umich.edu/~mejn/cp/programs.html
https://www.azimuthproject.org/azimuth/show/Stochastic+Hopf+bifurcation+in+Python
http://systems-sciences.uni-graz.at/etextbook/sw3/bifurcation.html

list of dynamical systems
https://en.wikipedia.org/wiki/List_of_dynamical_systems_and_differential_equations_topics
"""


##############################################

###### External Emulators Interface ##########
##############################################
# TODO: interface with, e.g., OpenAI gym



##########################################################
###### Base Control Profiles for System excitation #######
##########################################################
# TODO: functions generating baseline control signals or noise used for exciting the system for system ID and RL



def WhiteNoise(nx=1, nsim=100, xmax=1, xmin=0):
    """
    White Noise
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = xmin[k] + (xmax[k] - xmin[k])*np.random.rand(nsim)
        Signal.append(signal)
    return np.asarray(Signal).T

def Step(nx=1, nsim=100, tstep = 50, xmax=1, xmin=0):
    """
    step change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param tstep: (int) time of the step
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = np.ones(nsim)
        signal[0:tstep] = xmin[k]
        signal[tstep:] = xmax[k]
        Signal.append(signal)
    return np.asarray(Signal).T

def Ramp():
    """
    ramp change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    """
    pass

def Periodic(nx=1, nsim=100, numPeriods=1, xmax=1, xmin=0, form='sin'):
    """
    periodic signals, sine, cosine
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param numPeriods: (int) Number of periods
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param form: (str) form of the periodic signal 'sin' or 'cos'
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]).ravel()

    samples_period = nsim// numPeriods
    leftover = nsim % numPeriods
    Signal = []
    for k in range(nx):
        if form == 'sin':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        elif form == 'cos':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        signal = np.tile(base, numPeriods)
        signal = np.append(signal, base[0:leftover])
        Signal.append(signal)
    return np.asarray(Signal).T


# TODO: wrapper for scipy signal functions
# https://docs.scipy.org/doc/scipy/reference/signal.html#module-scipy.signal

def SignalComposite():
    """
    composite of signal excitations
    allows generating heterogenous signals
    """
    pass

def SignalSeries():
    """
    series of signal excitations
    allows combining sequence of different signals
    """
    pass


if __name__ == '__main__':
    """
    Tests
    """
    ninit = 0
    nsim = 1000

    building = Building_hf()   # instantiate building class
    building.parameters()      # load model parameters

    # generate input data
    M_flow = Periodic(nx=building.n_mf, nsim=nsim, numPeriods=6, xmax=building.mf_max, xmin=building.mf_min, form='sin')
    DT = Periodic(nx=building.n_dT, nsim=nsim, numPeriods=9, xmax=building.dT_max, xmin=building.dT_min, form='cos')
    D = building.D[ninit:nsim,:]

    # simulate open loop building
    U, X, Y = building.simulate(ninit, nsim, M_flow, DT, D)
    # plot trajectories
    plot.pltOL(Y, U, D, X)

