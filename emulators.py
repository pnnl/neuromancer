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
from scipy.integrate import odeint

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



class Tank(EmulatorBase):
    """
    Single Tank model
    original code obtained from APMonitor:
    https://apmonitor.com/pdc/index.php/Main/TankLevel

    TODO: linearized option
    https://apmonitor.com/pdc/index.php/Main/ModelLinearization
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        self.rho = 1000.0  # water density (kg/m^3)
        self.A = 1.0  # tank area (m^2)
        self.c1 = 80.0  # inlet valve coefficient (kg/s / %open)
        self.c2 = 40.0  # outlet valve coefficient (kg/s / %open)
        # Initial Conditions for the States
        self.x0 = 0
        # initial valve position
        self.u0 = 10

    # equations defining the dynamical system
    def equations(self, x, t, pump, valve):
        # States (1): level in the tanks
        # Inputs (1): valve
        # equations
        dx_dt = (self.c1/(self.rho*self.A)) *(1.0 - valve) * pump - self.c2 * np.sqrt(x)
        if x >= 1.0 and dx_dt > 0.0:
            dx_dt = 0
        return dx_dt

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, ts, Pump, Valve, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param Valve: (float) control input vector
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: The response matrices, i.e. X
        """
        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim) * ts + ninit
        X = []
        N = 0
        for pump, valve in zip(Pump, Valve):
            u = (pump, valve)
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=u)
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X)



class TwoTank(EmulatorBase):
    """
    Two Tank model
    original code obtained from APMonitor:
    https://apmonitor.com/do/index.php/Main/LevelControl

    TODO: linearized option
    https://apmonitor.com/pdc/index.php/Main/ModelLinearization
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        self.c1 = 0.08  # inlet valve coefficient
        self.c2 = 0.04  # tank outlet coefficient
        # Initial Conditions for the States
        self.x0 = [0, 0]

    # equations defining the dynamical system
    def equations(self, x, t, pump, valve):
        # States (2): level in the tanks
        h1 = x[0]
        h2 = x[1]
        # Inputs (2): pump and valve
        # pump = u[0]
        # valve = u[1]
        # equations
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * np.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * np.sqrt(h1) - self.c2 * np.sqrt(h2)
        if h1 >= 1.0 and dhdt1 > 0.0:
            dhdt1 = 0
        if h2 >= 1.0 and dhdt2 > 0.0:
            dhdt2 = 0
        dhdt = [dhdt1, dhdt2]
        return dhdt

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, ts, Pump, Valve, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param Pump: (float) control input vector
        :param Valve: (float) control input vector
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: The response matrices, i.e. X
        """
        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim) * ts + ninit
        X = []
        N = 0
        for pump, valve in zip(Pump, Valve):
            u = (pump, valve)
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=u)
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X)


class CSTR(EmulatorBase):
    """
    CSTR model
    original code obtained from APMonitor:
    http://apmonitor.com/do/index.php/Main/NonlinearControl
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        # Volumetric Flowrate (m^3/sec)
        self.q = 100
        # Volume of CSTR (m^3)
        self.V = 100
        # Density of A-B Mixture (kg/m^3)
        self.rho = 1000
        # Heat capacity of A-B Mixture (J/kg-K)
        self.Cp = 0.239
        # Heat of reaction for A->B (J/mol)
        self.mdelH = 5e4
        # E - Activation energy in the Arrhenius Equation (J/mol)
        # R - Universal Gas Constant = 8.31451 J/mol-K
        self.EoverR = 8750
        # Pre-exponential factor (1/sec)
        self.k0 = 7.2e10
        # U - Overall Heat Transfer Coefficient (W/m^2-K)
        # A - Area - this value is specific for the U calculation (m^2)
        self.UA = 5e4

        # Steady State Initial Conditions for the States
        self.Ca_ss = 0.87725294608097
        self.T_ss = 324.475443431599
        self.x0 = np.empty(2)
        self.x0[0] = self.Ca_ss
        self.x0[1] = self.T_ss

        # Steady State Initial Condition for the Uncontrolled Inputs
        self.u_ss = 300.0 # cooling jacket Temperature (K)
        self.Tf = 350 # Feed Temperature (K)
        self.Caf = 1 # Feed Concentration (mol/m^3)
        self.d0 = np.empty(2)
        self.d0[0] = self.Caf
        self.d0[1] = self.Tf

        self.nx = 2
        self.nu = 1
        self.nd = 2

    # equations defining the dynamical system
    def equations(self,x,t,u,Tf,Caf):
        # Inputs (1):
        # Temperature of cooling jacket (K)
        Tc = u

        # Disturbances (2):
        # Tf = Feed Temperature (K)
        # Caf = Feed Concentration (mol/m^3)

        # States (2):
        # Concentration of A in CSTR (mol/m^3)
        Ca = x[0]
        # Temperature in CSTR (K)
        T = x[1]

        # reaction rate
        rA = self.k0 * np.exp(-self.EoverR / T) * Ca
        # Calculate concentration derivative
        dCadt = self.q / self.V * (Caf - Ca) - rA
        # Calculate temperature derivative
        dTdt = self.q / self.V * (Tf - T) \
               + self.mdelH / (self.rho * self.Cp) * rA \
               + self.UA / self.V / self.rho / self.Cp * (Tc - T)
        xdot = np.zeros(2)
        xdot[0] = dCadt
        xdot[1] = dTdt
        return xdot

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, ts, U, x0=None, d0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param U: (float) control input vector
        :param x0: (float) state initial conditions
        :param d0: (float) uncontrolled input initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: The response matrices, i.e. X for states
        """
        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        if d0 is None:
            d0 = self.d0
        else:
            assert d0.shape[0] == self.nd, "Mismatch in u0 size"
        Caf = d0[0]
        Tf = d0[1]

        # time interval
        t = np.arange(0, nsim) * ts + ninit

        X = []
        N = 0
        for u in U:
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=(u, Tf, Caf))
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X)


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

# TODO: generate multiple files, make just one building_thermal model
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
    plot.pltOL(Y=Y, U=U, D=D, X=X)

    #   CSTR
    ninit = 0
    nsim = 251
    ts = 0.1
    # Step cooling temperature to 295
    u_ss = 300.0
    U = np.ones(nsim-1) * u_ss
    U[10:100] = 303.0
    U[100:190] = 297.0
    U[190:] = 300.0
    cstr_model = CSTR()  # instantiate CSTR class
    cstr_model.parameters()  # load model parameters
    # simulate open loop building
    X = cstr_model.simulate(ninit, nsim, ts, U)
    # plot trajectories
    plot.pltOL(Y=X[:,0], U=U, X=X[:,1])

    #   TwoTank
    ninit = 0
    nsim = 1001
    ts = 0.1
    # Inputs that can be adjusted
    pump = np.empty((nsim-1))
    pump[0] = 0
    pump[1:51] = 0.5
    pump[51:151] = 0.1
    pump[151:nsim-1] = 0.2
    valve = np.zeros((nsim-1))
    U = np.vstack([pump,valve]).T
    twotank_model = TwoTank()  # instantiate CSTR class
    twotank_model.parameters()  # load model parameters
    # simulate open loop building
    X = twotank_model.simulate(ninit, nsim, ts, pump, valve)
    # plot trajectories
    plot.pltOL(Y=X, U=U)

    # Tank
    ninit = 0
    nsim = 1001
    ts = 0.1
    # Inputs that can be adjusted
    pump = np.empty((nsim - 1))
    pump[0] = 0
    pump[1:51] = 0.5
    pump[51:151] = 0.1
    pump[151:nsim - 1] = 0.2
    valve = np.zeros((nsim - 1))
    U = np.vstack([pump, valve]).T
    tank_model = Tank()  # instantiate CSTR class
    tank_model.parameters()  # load model parameters
    # simulate open loop building
    # TODO: errors
    # X = tank_model.simulate(ninit, nsim, ts, pump, valve)
    # # plot trajectories
    # plot.pltOL(Y=X, U=valve)