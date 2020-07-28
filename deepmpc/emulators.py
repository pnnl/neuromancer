"""
wrapper for emulator dynamical models
Internal Emulators - in house ground truth equations
External Emulators - third party models
"""

from scipy.io import loadmat
# from scipy import signal
from abc import ABC, abstractmethod
import numpy as np
# import numdifftools as nd
import plot
from scipy.integrate import odeint
import gym
import control

####################################
###### Internal Emulators ##########
####################################

# data['R'] = emulators.Periodic(nx=data['Y'].shape[1], nsim=data['Y'].shape[0],
#                                numPeriods=np.ceil(data['Y'].shape[0] / 100).astype(int),
#                                xmax=0, xmin=1, form='sin')

"""
Base Classes
"""

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



class SSM(EmulatorBase):
    """
    base class state space model
    """
    def __init__(self):
        super().__init__()

    def parameters(self):
        self.ninit = 0
        self.nsim = 1000
        # steady state values
        self.x_ss = 0
        self.y_ss = 0

    def simulate(self, ninit=None, nsim=None, U=None, D=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) control signals
        :param D: (ndarray, shape=(nsim, self.nd)) measured disturbance signals
        :param x: (ndarray, shape=(self.nx)) Initial state.
        :return: The response matrices, i.e. X, Y, U, D
        """
        # default simulation setup parameters
        if ninit is None:
            ninit = self.ninit
            # warnings.warn('ninit was not defined, using default simulation setup')
        if nsim is None:
            nsim = self.nsim
            # warnings.warn('nsim was not defined, using default simulation setup')
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        if D is None:
            D = self.D[ninit: ninit + nsim, :] if self.D is not None else None
        if U is None:
            U = self.U[ninit: ninit + nsim, :] if self.U is not None else None
            # warnings.warn('U was not defined, using default trajectories')
        X, Y = [], []
        for k in range(nsim):
            u = U[k, :] if U is not None else None
            d = D[k, :] if D is not None else None
            x, y = self.equations(x, u, d)
            X.append(x + self.x_ss)
            Y.append(y - self.y_ss)
        Xout = np.asarray(X).reshape(nsim, self.nx)
        Yout = np.asarray(Y).reshape(nsim, self.ny)
        Uout = np.asarray(U).reshape(nsim, self.nu) if U is not None else None
        Dout = np.asarray(D).reshape(nsim, self.nd) if D is not None else None
        return Xout, Yout, Uout, Dout


class ODE_Autonomous(EmulatorBase):
    """
    base class autonomous ODE
    """
    def __init__(self):
        super().__init__()

    def parameters(self):
        self.ninit = 0
        self.nsim = 1000
        self.ts = 0.1

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit=None, nsim=None, ts=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: The response matrices, i.e. X
        """

        # default simulation setup parameters
        if ninit is None:
            ninit = self.ninit
            # warnings.warn('ninit was not defined, using default simulation setup')
        if nsim is None:
            nsim = self.nsim
            # warnings.warn('nsim was not defined, using default simulation setup')
        if ts is None:
            ts = self.ts
            # warnings.warn('ts was not defined, using default simulation setup')

        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim+1) * ts + ninit
        X = []
        for N in range(nsim-1):
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT)
            x = xdot[-1]
            X.append(x)  # updated states trajectories
        Xout = np.asarray(X)
        Yout = np.asarray(X)
        Uout = None
        Dout = None
        return Xout, Yout, Uout, Dout


class ODE_NonAutonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    def __init__(self):
        super().__init__()

    def parameters(self):
        self.ninit = 0
        self.nsim = 1000
        self.ts = 0.1

    # N-step forward simulation of the dynamical system
    def simulate(self, U=None, ninit=None, nsim=None, ts=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: X, Y, U, D
        """

        # default simulation setup parameters
        if ninit is None:
            ninit = self.ninit
            # warnings.warn('ninit was not defined, using default simulation setup')
        if nsim is None:
            nsim = self.nsim
            # warnings.warn('nsim was not defined, using default simulation setup')
        if ts is None:
            ts = self.ts
            # warnings.warn('ts was not defined, using default simulation setup')
        if U is None:
            U = self.U
            # warnings.warn('U was not defined, using default trajectories')

        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim+1) * ts + ninit
        X = []
        N = 0
        for u in U:
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=(u,))
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(X)
        Uout = np.asarray(U)
        Dout = None
        return Xout, Yout, Uout, Dout


"""
Linear ODEs and SSMs
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


"""
Hybrid linear ODEs
CartPole, bauncing ball
"""

class LinCartPole(EmulatorBase):
    """
    Linearized Hybrid model of Inverted pendulum
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    TODO: nonlinear case
    https://apmonitor.com/do/index.php/Main/InvertedPendulum
    """
    def __init__(self):
        super().__init__()
        pass

    def parameters(self, Ts=0.1):
        self.M = 0.5
        self.m = 0.2
        self.b = 0.1
        self.I = 0.006
        self.g = 9.8
        self.l = 0.3
        self.p = self.I*(self.M+self.m)+self.M*self.m*self.l**2; # denominator for the A and B matrices
        # self.dumping = 0.1
        self.theta1 = -(self.I+self.m*self.l**2)*self.b/self.p
        self.theta2 = (self.m**2*self.g*self.l**2)/self.p
        self.theta3 = -(self.m*self.l*self.b)/self.p
        self.theta4 = (self.m*self.g*self.l*(self.M+self.m)/self.p)
        self.A = np.asarray([[0,1,0,0],[0,self.theta1,self.theta2,0],
                             [0,0,0,1],[0,self.theta3,self.theta4,0]])
        self.B = np.asarray([[0],[(self.I+self.m*self.l**2)/self.p],
                            [0],[self.m*self.l/self.p]])
        self.C = np.asarray([[1,0,0,0],[0,0,1,0]])
        self.D = np.asarray([[0],[0]])
        self.ssm = control.StateSpace(self.A, self.B, self.C, self.D)
        self.Ts = Ts
        self.ssmd = self.ssm.sample(self.Ts, method='euler')

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.x0 = np.asarray([0,0,-1,0])

    # equations defining single step of the dynamical system
    def equations(self, x, u):
        # Inputs (1): u is the force applied to the cart
        # States (4):
        # x1 position of the cart,
        # x2 velocity of the cart,
        # x3 angle of the pendulum relative to the cart
        # x4 rate of angle change
        x = np.matmul(np.asarray(self.ssmd.A), x) + np.matmul(np.asarray(self.ssmd.B), u).T
        #  physical constraints: position between +-10
        if x[0] >= 10:
            x[0] = 10
            x[1] = 0
        if x[0] <= -10:
            x[0] = -10
            x[1] = 0
        # angle in between +- 2*pi radians = -+ 360 degrees
        x[3] = np.mod(x[3], 2*np.pi)
        # positive +180 degrees and negative direction -180 degrees
        if x[3] >= np.pi:
            x[3] = np.pi-x[3]
        if x[3] <= -np.pi:
            x[3] = -np.pi-x[3]

        y = np.matmul(np.asarray(self.ssmd.C), x)
        return x, y

    def simulate(self, ninit, nsim, U, ts=None, x0=None):
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

        # if Ts is not None:
        #     self.parameters(Ts)

        X, Y = [], []
        N = 0
        for u in U:
            x, y = self.equations(x, u)
            X.append(x)  # updated states trajectories
            Y.append(y)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(Y)
        Uout = None
        Dout = None
        return Xout, Yout, Uout, Dout




"""
Nonlinear ODEs

https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
https://en.wikipedia.org/wiki/List_of_dynamical_systems_and_differential_equations_topics
"""


class UniversalOscillator(ODE_Autonomous):
    """
    Hharmonic oscillator
    https://en.wikipedia.org/wiki/Harmonic_oscillator
    https://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.mu = 2
        self.omega = 1
        self.x0 = [1.0, 0.0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 10001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = x[1]
        dx2 = -2*self.mu*x[1] - x[0] + np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx

# TODO: debug
class SEIR_population(EmulatorBase):
    """
    Susceptible, Exposed, Infected, and Recovered (SEIR) population population model
    COVID19 spread
    source of the model:
    https://apmonitor.com/do/index.php/Main/COVID-19Response

    states:
    Susceptible (s): population fraction that is susceptible to the virus
    Exposed (e): population fraction is infected with the virus but does not transmit to others
    Infectious (i): population fraction that is infected and can infect others
    Recovered (r): population fraction recovered from infection and is immune from further infection
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        self.N = 10000 # population
        # initial number of infected and recovered individuals
        self.e_0 = 1 / self.N
        self.i_0 = 0.00
        self.r_0 = 0.00
        self.s_0 = 1 - self.e_0 - self.i_0 - self.r_0
        self.x0 = np.asarray([self.s_0, self.e_0, self.i_0, self.r_0])

        self.nx = 4
        self.nu = 1

        self.t_incubation = 5.1
        self.t_infective = 3.3
        self.R0 = 2.4
        self.alpha = 1 / self.t_incubation
        self.gamma = 1 / self.t_infective
        self.beta = self.R0 * self.gamma

    # equations defining the dynamical system
    # def equations(self, x, u, alpha, beta, gamma):
    def equations(self, x, t, u):
        # Inputs (1): social distancing (u=0 (none), u=1 (total isolation))
        # States (4):
        # Susceptible (s): population fraction that is susceptible to the virus
        # Exposed (e): population fraction is infected with the virus but does not transmit to others
        # Infectious (i): population fraction that is infected and can infect others
        # Recovered (r): population fraction recovered from infection and is immune from further infection
        s = x[0]
        e = x[1]
        i = x[2]
        r = x[3]
        # SEIR equations
        sdt = -(1 - u) * self.beta * s * i,
        edt = (1 - u) * self.beta * s * i - self.alpha * e,
        idt = self.alpha * e - self.gamma * i,
        rdt = self.gamma * i
        dx = [sdt, edt, idt, rdt]
        # dx = np.asarray([sdt, edt, idt, rdt])
        return dx

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, ts, U, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param U: (float) control input vector (social distancing)
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states (SEIR)
        :return: The response matrices, i.e. X
        """
        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0

        # alpha = 1 / self.t_incubation
        # gamma = 1 / self.t_infective
        # beta = self.R0 * self.gamma

        # time interval
        t = np.arange(0, nsim) * ts + ninit
        X = []
        N = 0
        for u in U:
            dT = [t[N], t[N + 1]]
            # TODO: error here
            xdot = odeint(self.equations, x, dT, args=(u,))
            # xdot = odeint(self.equations, x, dT,
            #               args=(u, alpha, beta, gamma))
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X)


class Tank(EmulatorBase):
    """
    Single Tank model
    original code obtained from APMonitor:
    https://apmonitor.com/pdc/index.php/Main/TankLevel
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


class TwoTank(ODE_NonAutonomous):
    """
    Two Tank model
    original code obtained from APMonitor:
    https://apmonitor.com/do/index.php/Main/LevelControl
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.c1 = 0.08  # inlet valve coefficient
        self.c2 = 0.04  # tank outlet coefficient
        # Initial Conditions for the States
        self.x0 = np.asarray([0, 0])
        # default simulation setup
        self.ninit = 0
        self.nsim = 1001
        self.ts = 0.1
        pump = np.empty((self.nsim - 1))
        pump[0] = 0
        pump[1:501] = 0.5
        pump[251:551] = 0.1
        pump[551: self.nsim - 1] = 0.2
        valve = np.zeros((self.nsim - 1))
        self.U = np.vstack([pump, valve]).T
        self.nu = 2
        self.nx = 2

    # equations defining the dynamical system
    def equations(self, x, t, u):
        # States (2): level in the tanks
        h1 = x[0]
        h2 = x[1]
        # Inputs (2): pump and valve
        pump = u[0]
        valve = u[1]
        # equations
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * np.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * np.sqrt(h1) - self.c2 * np.sqrt(h2)
        if h1 >= 1.0 and dhdt1 > 0.0:
            dhdt1 = 0
        if h2 >= 1.0 and dhdt2 > 0.0:
            dhdt2 = 0
        dhdt = [dhdt1, dhdt2]
        return dhdt


class CSTR(ODE_NonAutonomous):
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
        # dimensions
        self.nx = 2
        self.nu = 1
        self.nd = 2
        # default simulation setup
        self.ninit = 0
        self.nsim = 1001
        self.ts = 0.1
        # Step cooling temperature to 295
        u_ss = 300.0
        U = np.ones(self.nsim - 1) * u_ss
        U[50:150] = 303.0
        U[150:250] = 297.0
        U[250:350] = 295.0
        U[350:450] = 298.0
        U[450:550] = 298.0
        U[650:750] = 303.0
        U[750:850] = 306.0
        U[850:] = 300.0
        self.U =U

    # equations defining the dynamical system
    def equations(self,x,t,u):
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
        dCadt = self.q / self.V * (self.Caf - Ca) - rA
        # Calculate temperature derivative
        dTdt = self.q / self.V * (self.Tf - T) \
               + self.mdelH / (self.rho * self.Cp) * rA \
               + self.UA / self.V / self.rho / self.Cp * (Tc - T)
        xdot = np.zeros(2)
        xdot[0] = dCadt
        xdot[1] = dTdt
        return xdot


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
        self.nsim = 1000

    # equations defining single step of the dynamical system
    def equations(self, x):
        pass

    def simulate(self, ninit=None, nsim=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if nsim is None:
            nsim = self.nsim
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


class BuildingEnvelope(SSM):
    """
    building envelope heat transfer model
    linear building envelope dynamics and bilinear heat flow input dynamics
    different building types are stored in ./emulators/buildings/*.mat
    models obtained from: https://github.com/drgona/BeSim
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self, system='Reno_full', linear=True):
        # file paths for different building models
        systems = {'SimpleSingleZone': './emulators/buildings/SimpleSingleZone.mat',
                   'Reno_full': './emulators/buildings/Reno_full.mat',
                   'Reno_ROM40': './emulators/buildings/Reno_ROM40.mat',
                   'RenoLight_full': './emulators/buildings/RenoLight_full.mat',
                   'RenoLight_ROM40': './emulators/buildings/RenoLight_ROM40.mat',
                   'Old_full': './emulators/buildings/Old_full.mat',
                   'Old_ROM40': './emulators/buildings/Old_ROM40.mat',
                   'HollandschHuys_full': './emulators/buildings/HollandschHuys_full.mat',
                   'HollandschHuys_ROM100': './emulators/buildings/HollandschHuys_ROM100.mat',
                   'Infrax_full': './emulators/buildings/Infrax_full.mat',
                   'Infrax_ROM100': './emulators/buildings/Infrax_ROM100.mat'
                   }

        file_path = systems[system]
        file = loadmat(file_path)
        self.system = system
        self.linear = linear  # if True use only linear building envelope model with Q as U
        #  LTI SSM model
        self.A = file['Ad']
        self.B = file['Bd']
        self.C = file['Cd']
        self.E = file['Ed']
        self.G = file['Gd']
        self.F = file['Fd']
        #  constraints bounds
        self.ts = file['Ts']  # sampling time TODO: not correct value for some building models
        # self.TSup = file['TSup']  # supply temperature
        self.umax = file['umax'].squeeze()  # max heat per zone
        self.umin = file['umin'].squeeze() # min heat per zone
        if not self.linear:
            self.dT_max = file['dT_max']  # maximal temperature difference deg C
            self.dT_min = file['dT_min']  # minimal temperature difference deg C
            self.mf_max = file['mf_max'].squeeze()  # maximal nominal mass flow l/h
            self.mf_min = file['mf_min'].squeeze()  # minimal nominal mass flow l/h
            #   heat flow equation constants
            self.rho = 0.997  # density  of water kg/1l
            self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
            self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds
            # building type
        self.type = file['type']
        self.HC_system = file['HC_system']
        # problem dimensions
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nq = self.B.shape[1]
        self.nd = self.E.shape[1]
        if self.linear:
            self.nu = self.nq
        else:
            self.n_mf = self.B.shape[1]
            self.n_dT = self.dT_max.shape[0]
            self.nu = self.n_mf + self.n_dT
        # initial conditions and disturbance profiles
        if self.system == 'SimpleSingleZone':
            self.x0 = file['x0'].reshape(self.nx)
        else:
            self.x0 = 0*np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.D = file['disturb'] # pre-defined disturbance profiles
        #  steady states - linearization offsets
        self.x_ss = file['x_ss']
        self.y_ss = file['y_ss']
        # default simulation setup
        self.ninit = 0
        self.nsim = np.min([8640, self.D.shape[0]])
        if self.linear:
            self.U = Periodic(nx=self.nu, nsim=self.nsim, numPeriods=21, xmax=self.umax/2, xmin=self.umin, form='sin')
        else:
            self.M_flow = self.mf_max/2+RandomWalk(nx=self.n_mf, nsim=self.nsim, xmax=self.mf_max/2, xmin=self.mf_min, sigma=0.05)
            # self.M_flow = Periodic(nx=self.n_mf, nsim=self.nsim, numPeriods=21, xmax=self.mf_max, xmin=self.mf_min, form='sin')
            # self.DT = Periodic(nx=self.n_dT, nsim=self.nsim, numPeriods=15, xmax=self.dT_max/2, xmin=self.dT_min, form='cos')
            self.DT = RandomWalk(nx=self.n_dT, nsim=self.nsim, xmax=self.dT_max*0.6, xmin=self.dT_min, sigma=0.05)
            self.U = np.hstack([self.M_flow, self.DT])

    # equations defining single step of the dynamical system
    def equations(self, x, u, d):
        if self.linear:
            q = u
        else:
            m_flow = u[0:self.n_mf]
            dT = u[self.n_mf:self.n_mf+self.n_dT]
            q = m_flow * self.rho * self.cp * self.time_reg * dT
        x = np.matmul(self.A, x) + np.matmul(self.B, q) + np.matmul(self.E, d) + self.G.ravel()
        y = np.matmul(self.C, x) + self.F.ravel()
        return x, y



"""
Linear PDEs
"""



"""
Nonlinear PDEs

https://en.wikipedia.org/wiki/List_of_nonlinear_partial_differential_equations
"""



"""
Chaotic nonlinear ODEs 

https://en.wikipedia.org/wiki/List_of_chaotic_maps
"""


class Lorenz96(ODE_Autonomous):
    """
    Lorenz 96 model
    https://en.wikipedia.org/wiki/Lorenz_96_model
    """
    def __init__(self):
        super().__init__()
        pass

    # parameters of the dynamical system
    def parameters(self):
        super().parameters() # inherit parameters of the mothership
        self.N = 36  # Number of variables
        self.F = 8  # Forcing
        self.x0 = self.F*np.ones(self.N)
        self.x0[19] += 0.01  # Add small perturbation to random variable
        # self.x0[np.random.randint(0, self.N)] += 0.01  # Add small perturbation to random variable
        # default simulation setup
        self.ninit = 0
        self.nsim = 5001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        # Compute state derivatives
        dx = np.zeros(self.N)
        # First the 3 edge cases: i=1,2,N
        dx[0] = (x[1] - x[self.N - 2]) * x[self.N - 1] - x[0]
        dx[1] = (x[2] - x[self.N - 1]) * x[0] - x[1]
        dx[self.N - 1] = (x[0] - x[self.N - 3]) * x[self.N - 2] - x[self.N - 1]
        # Then the general case
        for i in range(2, self.N - 1):
            dx[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
        # Add the forcing term
        dx = dx + self.F
        return dx


class LorenzSystem(ODE_Autonomous):
    """
    Lorenz System
    https://en.wikipedia.org/wiki/Lorenz_system#Analysis
    # https://ipywidgets.readthedocs.io/en/stable/examples/Lorenz%20Differential%20Equations.html
    # https://scipython.com/blog/the-lorenz-attractor/
    # https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.x0 = [1.0, 1.0, 1.0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 5001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = self.sigma*(x[1] - x[0])
        dx2 = x[0]*(self.rho - x[2]) - x[1]
        dx3 = x[0]*x[1] - self.beta*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class VanDerPol(ODE_Autonomous):
    """
    Van der Pol oscillator
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.mu = 1.0
        self.x0 = [1, 2]
        # default simulation setup
        self.ninit = 0
        self.nsim = 401
        self.ts = 0.1

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = self.mu*(x[0] - 1./3.*x[0]**3 - x[1])
        dx2= x[0]/self.mu
        dx = [dx1, dx2]
        return dx


class HindmarshRose(ODE_NonAutonomous):
    """
    Hindmarsh–Rose model of neuronal activity
    https://en.wikipedia.org/wiki/Hindmarsh%E2%80%93Rose_model
    https://demonstrations.wolfram.com/HindmarshRoseNeuronModel/
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.a = 1
        self.b = 2.6
        self.c = 1
        self.d = 5
        self.s = 4
        self.xR = -8/5
        self.r = 0.01
        self.umin = -10
        self.umax = 10
        self.x0 = np.asarray([-5,-10,0])
        # default simulation setup
        self.ninit = 0
        self.nsim = 5001
        self.ts = 0.1
        self.U = 3 * np.asarray([np.ones((self.nsim - 1))]).T
        self.nu = 1
        self.nx = 3

    # equations defining the dynamical system
    def equations(self, x, t, u):
        # Derivatives
        theta = -self.a*x[0]**3 + self.b*x[0]**2
        phi = self.c -self.d*x[0]**2
        dx1 = x[1] + theta - x[2] + u
        dx2 = phi - x[1]
        dx3 = self.r*(self.s*(x[0]-self.xR)-x[2])
        dx = [dx1, dx2, dx3]
        return dx


class ThomasAttractor(ODE_Autonomous):
    """
    Thomas' cyclically symmetric attractor
    https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.b = 0.208186
        self.x0 = [1,-1,1]
        # default simulation setup
        self.ninit = 0
        self.nsim = 5001
        self.ts = 0.1

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = np.sin(x[1]) - self.b*x[0]
        dx2 = np.sin(x[2]) - self.b*x[1]
        dx3 = np.sin(x[0]) - self.b*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class RosslerAttractor(ODE_Autonomous):
    """
    Rössler attractor
    https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.x0 = [0,0,0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 20001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = - x[1] - x[2]
        dx2 = x[0] + self.a*x[1]
        dx3 = self.b + x[2]*(x[0]-self.c)
        dx = [dx1, dx2, dx3]
        return dx


class LotkaVolterra(ODE_Autonomous):
    """
    Lotka–Volterra equations, also known as the predator–prey equations
    https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.a = 1.
        self.b = 0.1
        self.c = 1.5
        self.d = 0.75
        self.x0 = [5, 100]
        # default simulation setup
        self.ninit = 0
        self.nsim = 2001
        self.ts = 0.1

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = self.a*x[0] - self.b*x[0]*x[1]
        dx2 = -self.c*x[1] + self.d*self.b*x[0]*x[1]
        dx = [dx1, dx2]
        return dx


class Brusselator1D(ODE_Autonomous):
    """
    Brusselator
    https://en.wikipedia.org/wiki/Brusselator
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.a = 1.0
        self.b = 3.0
        self.x0 = [1.0, 1.0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 501
        self.ts = 0.1

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = self.a + x[1]*x[0]**2 -self.b*x[0] - x[0]
        dx2 = self.b*x[0] - x[1]*x[0]**2
        dx = [dx1, dx2]
        return dx


class ChuaCircuit(ODE_Autonomous):
    """
    Chua's circuit
    https://en.wikipedia.org/wiki/Chua%27s_circuit
    https://www.chuacircuits.com/matlabsim.php
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.a = 15.6
        self.b = 28.0
        # self.R = 1.0
        # self.C = 1.0
        self.m0 = -1.143
        self.m1 = -0.714
        self.x0 = [0.7, 0.0, 0.0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 10001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        fx = self.m1*x[0] + 0.5*(self.m0 - self.m1)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        # Derivatives
        dx1 = self.a*(x[1] - x[0] - fx)
        dx2 = x[0] - x[1] + x[2]
        dx3 = -self.b*x[1]
        dx = [dx1, dx2, dx3]
        return dx


class Duffing(ODE_Autonomous):
    """
    Duffing equation
    https://en.wikipedia.org/wiki/Duffing_equation
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self):
        super().parameters()
        self.delta = 0.02
        self.alpha = 1
        self.beta = 5
        self.gamma = 8
        self.omega = 0.5
        self.x0 = [1.0, 0.0]
        # default simulation setup
        self.ninit = 0
        self.nsim = 10001
        self.ts = 0.01

    # equations defining the dynamical system
    def equations(self, x, t):
        # Derivatives
        dx1 = x[1]
        dx2 = - self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3 + self.gamma*np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx




"""
Chaotic nonlinear PDEs
"""



"""
Cellular automata
"""



"""
Fractals
"""
class Mandelbrot(EmulatorBase):
    """
    IDEA: use mandelbrot zoom video as our dataset for training
    Cool effect
    """
    def __init__(self):
        super().__init__()
        pass




##############################################

###### External Emulators Interface ##########
##############################################

"""
# OpenAI gym wrapper
"""

class GymWrapper(EmulatorBase):
    """
    wrapper for OpenAI gym environments
    https://gym.openai.com/read-only.html
    https://github.com/openai/gym
    """
    def __init__(self):
        super().__init__()

    # parameters of the dynamical system
    def parameters(self, system='CartPole-v1'):
        self.system = system
        self.env = gym.make(self.system)
        self.env.reset()  # to reset the environment state
        self.x0 = self.env.state
        self.nx = self.x0.shape[0]
        self.action_sample = self.env.action_space.sample()
        self.nu = np.asarray([self.action_sample]).shape[0]
        #     default simulation setup
        self.nsim = 501
        self.U = np.zeros([(self.nsim - 1), self.nu])
        if type(self.action_sample) == int:
            self.U = self.U.astype(int)

    # equations defining the dynamical system
    def equations(self, x, u):
        if type(self.action_sample) == int:
            u = u.item()
        self.env.state = x
        x, reward, done, info = self.env.step(u)
        return x, reward

    def simulate(self, nsim=None, U=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(self.nu)) control actions
        :param x0: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if nsim is None:
            nsim = self.nsim
        if U is None:
            U = self.U
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0

        X, Reward = [], []
        N = 0
        for u in U:
            x, reward = self.equations(x, u)
            X.append(x)  # updated states trajectories
            Reward.append(reward)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(Reward).reshape(-1,1)
        Uout = np.asarray(U)
        Dout = None
        return Xout, Yout, Uout, Dout



##########################################################
###### Base Control Profiles for System excitation #######
##########################################################

def RandomWalk(nx=1, nsim=100, xmax=1, xmin=0, sigma=0.05):
    """

    :param nx:
    :param nsim:
    :param xmax:
    :param xmin:
    :return:
    """
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
        xmax = np.asarray([xmax]*nx).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]*nx).ravel()

    xmax = xmax.reshape(nx)
    xmin = xmin.reshape(nx)

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


systems = {# non-autonomous ODEs
           'CSTR': CSTR,
           'TwoTank': TwoTank,
           # autonomous chaotic ODEs
           'LorenzSystem': LorenzSystem,
           'Lorenz96': Lorenz96,
           'VanDerPol': VanDerPol,
           'ThomasAttractor': ThomasAttractor,
           'RosslerAttractor': RosslerAttractor,
           'LotkaVolterra': LotkaVolterra,
           'Brusselator1D': Brusselator1D,
           'ChuaCircuit': ChuaCircuit,
           'Duffing': Duffing,
           'UniversalOscillator': UniversalOscillator,
           # non-autonomous chaotic ODEs
           'HindmarshRose': HindmarshRose,
           # OpenAI gym environments
           'Pendulum-v0': GymWrapper,
           'CartPole-v1': GymWrapper,
           'Acrobot-v1': GymWrapper,
           'MountainCar-v0': GymWrapper,
           'MountainCarContinuous-v0': GymWrapper,
           # partially observable building state space models with external disturbances
           'SimpleSingleZone': BuildingEnvelope,
           'Reno_full': BuildingEnvelope,
           'Reno_ROM40': BuildingEnvelope,
           'RenoLight_full': BuildingEnvelope,
           'RenoLight_ROM40': BuildingEnvelope,
           'Old_full': BuildingEnvelope,
           'Old_ROM40': BuildingEnvelope,
           'HollandschHuys_full': BuildingEnvelope,
           'HollandschHuys_ROM100': BuildingEnvelope,
           'Infrax_full': BuildingEnvelope,
           'Infrax_ROM100': BuildingEnvelope
           }



if __name__ == '__main__':
    """
    Tests
    """

    # building model
    ninit = 0
    building = BuildingEnvelope()   # instantiate building class
    building.parameters(system='HollandschHuys_full', linear=False)      # load model parameters
    # simulate open loop building
    X, Y, U, D= building.simulate(ninit=ninit)
    # plot trajectories
    plot.pltOL(Y=Y, U=U, D=D, X=X)
    plot.pltPhase(X=Y)

    #   CSTR
    cstr_model = CSTR()  # instantiate CSTR class
    cstr_model.parameters()  # load model parameters
    X, Y, U, D = cstr_model.simulate() # simulate open loop
    plot.pltOL(Y=X[:,0], U=cstr_model.U, X=X[:,1]) # plot trajectories
    plot.pltPhase(X=X)

    #   TwoTank
    twotank_model = TwoTank()  # instantiate model class
    twotank_model.parameters()  # load model parameters
    X, Y, U, D = twotank_model.simulate() # simulate open loop
    # X = twotank_model.simulate(ninit=ninit, nsim=nsim, ts=ts, U=U) #  example custom simulation setup
    # plot trajectories
    plot.pltOL(Y=X, U=U)
    plot.pltPhase(X=X)

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
    tank_model = Tank()  # instantiate model class
    tank_model.parameters()  # load model parameters
    # simulate open loop
    # TODO: errors
    # X = tank_model.simulate(ninit, nsim, ts, pump, valve)
    # # plot trajectories
    # plot.pltOL(Y=X, U=valve)

    #  SEIR
    ninit = 0
    nsim = 201
    ts = 1
    # Inputs that can be adjusted
    U = np.asarray([np.zeros((nsim - 1))]).T
    seir_model = SEIR_population()  # instantiate model class
    seir_model.parameters()  # load model parameters
    # simulate open loop
    # X = seir_model.simulate(ninit, nsim, ts, U)
    # # plot trajectories
    # plot.pltOL(Y=X, U=U)

    # linear cart pole system
    ninit = 0
    nsim = 201
    ts = 0.1
    U = np.asarray([np.zeros((nsim - 1))]).T
    #  inverted pendulum
    lcp_model = LinCartPole()  # instantiate model class
    lcp_model.parameters()
    # simulate open loop
    X, Y, U, D = lcp_model.simulate(ninit=ninit, nsim=nsim, ts=ts, U=U)
    # plot trajectories
    plot.pltOL(Y=Y, X=X, U=U)
    plot.pltPhase(X=X)

    # TODO: double check dimensions of x
    # OpenAi gym environment wrapper
    # tested envs:
    #   1, Classical: 'Pendulum-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
    #                          'MountainCarContinuous-v0'
    environment = 'Pendulum-v0'
    gym_model = GymWrapper()
    gym_model.parameters(system=environment)
    ninit = 0
    nsim = 201
    U = np.zeros([(nsim - 1), gym_model.nu])
    if type(gym_model.action_sample) == int:
        U = U.astype(int)
    X, Y, U, D = gym_model.simulate(ninit=ninit, nsim=nsim, U=U)
    # X, Reward, U = gym_model.simulate() # example with default setup
    plot.pltOL(Y=Y, X=X, U=U)
    plot.pltPhase(X=X)
    # TODO: include visualization option for the trajectories or render of OpenAI gym


    # Lorenz 96
    lorenz96_model = Lorenz96()  # instantiate model class
    lorenz96_model.parameters()
    X, Y, U, D = lorenz96_model.simulate() # simulate open loop
    plot.pltOL(Y=X) # plot trajectories
    plot.pltPhase(X=X) # phase plot

    # LorenzSystem
    lorenz_model = LorenzSystem()  # instantiate model class
    lorenz_model.parameters()
    X, Y, U, D = lorenz_model.simulate() # simulate open loop
    plot.pltOL(Y=X) # plot trajectories
    plot.pltPhase(X=X) # phase plot
    plot.pltRecurrence(X=X) # recurrence plots
    plot.pltCorrelate(X=X) # correlation plots

    # VanDerPol
    vdp_model = VanDerPol()  # instantiate model class
    vdp_model.parameters()   # instantiate parameters
    X, Y, U, D = vdp_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # HindmarshRose
    # example of non-autonomous system with default simulation setup
    HR_model = HindmarshRose()  # instantiate model class
    HR_model.parameters()
    X, Y, U, D = HR_model.simulate()
    plot.pltOL(Y=X, U=U)
    plot.pltPhase(X=X)

    # ThomasCSattractor
    nsim = 10000 #  example with default sim setup and custom nsim
    TCSA_model = ThomasAttractor()  # instantiate
    TCSA_model.parameters()
    X, Y, U, D = TCSA_model.simulate(nsim=nsim)
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # RösslerAttractor
    Ross_model = RosslerAttractor()  # instantiate model
    Ross_model.parameters()
    X, Y, U, D = Ross_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # LotkaVolterra
    lv_model = LotkaVolterra()  # instantiate model class
    lv_model.parameters()
    X, Y, U, D = lv_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # Brusselator1D
    bruss_model = Brusselator1D()  # instantiate model class
    bruss_model.parameters()
    X, Y, U, D = bruss_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # ChuaCircuit
    chua_model = ChuaCircuit()  # instantiate model class
    chua_model.parameters()
    X, Y, U, D = chua_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # Duffing
    Duffing_model = Duffing()  # instantiate model class
    Duffing_model.parameters()
    X, Y, U, D = Duffing_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # UniversalOscillator
    oscillator_model = UniversalOscillator()  # instantiate model class
    oscillator_model.parameters()
    X, Y, U, D = oscillator_model.simulate() # simulate open loop
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)


# TODO: generate meaningfull reference signals for each emulator
