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
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import gym
import control


####################################
###### Internal Emulators ##########
####################################

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


class ODE_Autonomous(EmulatorBase):
    """
    base class autonomous ODE
    """
    def __init__(self):
        super().__init__()

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, ts, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
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
        for N in range(nsim-1):
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT)
            x = xdot[-1]
            X.append(x)  # updated states trajectories
        return np.asarray(X)

class ODE_NonAutonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    def __init__(self):
        super().__init__()

    # N-step forward simulation of the dynamical system
    def simulate(self, ninit, nsim, U, ts=0.1, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
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
        for u in U:
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=(u,))
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X)



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


class SimpleHarmonicMotion(EmulatorBase):
    """
    https://en.wikipedia.org/wiki/Simple_harmonic_motion
    """
    def __init__(self):
        super().__init__()
        pass


"""
Hybrid linear ODEs
CartPole, bauncing ball
"""

class LinCartPole(EmulatorBase):
    """
    Linearized Hybrid model of Inverted pendulum
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    TODO: visualizations + nonlinear case
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
        return np.asarray(X), np.asarray(Y)




"""
Nonlinear ODEs
list of nlin ODEs
https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
"""




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

APmonitor PDEs
https://apmonitor.com/do/index.php/Main/PartialDifferentialEquations

Artificial Lift Rod Pump
https://apm.byu.edu/prism/index.php/Projects/HydraulicRodPumping
https://github.com/BYU-PRISM/USTAR-Artificial-Lift

fuel cell
https://apmonitor.com/do/index.php/Main/SolidOxideFuelCell
"""



"""
Chaotic nonlinear ODEs 

chaotic systems
https://en.wikipedia.org/wiki/List_of_chaotic_maps

TODO:
https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
# 2D case brusselator PDE
http://runge.math.smu.edu/ParallelComputing/_downloads/brusselator.pdf
https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/
http://math.colgate.edu/math329/
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
        self.N = 36  # Number of variables
        self.F = 8  # Forcing
        self.x0 = self.F*np.ones(self.N)
        self.x0[19] += 0.01  # Add small perturbation to random variable
        # self.x0[np.random.randint(0, self.N)] += 0.01  # Add small perturbation to random variable

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
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.x0 = [1.0, 1.0, 1.0]

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
        self.mu = 1.0
        self.x0 = [1, 2]

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
        self.a = 1
        self.b = 2.6
        self.c = 1
        self.d = 5
        self.s = 4
        self.xR = -8/5
        self.r = 0.01
        self.umin = -10
        self.umax = 10
        self.x0 = [-5,-10,0]

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
        self.b = 0.208186
        self.x0 = [1,-1,1]

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
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.x0 = [0,0,0]

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
        self.a = 1.
        self.b = 0.1
        self.c = 1.5
        self.d = 0.75
        self.x0 = [5, 100]

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
        self.a = 1.0
        self.b = 3.0
        self.x0 = [1.0, 1.0]

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
        self.a = 15.6
        self.b = 28.0
        # self.R = 1.0
        # self.C = 1.0
        self.m0 = -1.143
        self.m1 = -0.714
        self.x0 = [0.7, 0.0, 0.0]

    # equations defining the dynamical system
    def equations(self, x, t):
        fx = self.m1*x[0] + 0.5*(self.m0 - self.m1)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        # Derivatives
        dx1 = self.a*(x[1] - x[0] - fx)
        dx2 = x[0] - x[1] + x[2]
        dx3 = -self.b*x[1]
        dx = [dx1, dx2, dx3]
        return dx


"""
Chaotic nonlinear PDEs
"""



"""
Fractals
"""


class Mandelbrot(EmulatorBase):
    """
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

"""
# OpenAI gym wrapper

# examples:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.pyhttps://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

investigate potential third party environments
https://github.com/openai/gym/blob/master/docs/environments.md#third-party-environments

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
    def parameters(self, env_pick='CartPole-v1'):
        self.type = env_pick
        self.env = gym.make(self.type)
        self.env.reset() # to reset the environment state
        self.x0 = self.env.state
        self.nx = self.x0.shape[0]
        self.action_sample = self.env.action_space.sample()
        self.nu = np.asarray([self.action_sample]).shape[0]

    # equations defining the dynamical system
    def equations(self, x, u):
        if type(self.action_sample) == int:
            u = u.item()
        self.env.state = x
        x, reward, done, info = self.env.step(u)
        return x, reward

    def simulate(self, ninit, nsim, U, x0=None):
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

        X, Reward = [], []
        N = 0
        for u in U:
            x, reward = self.equations(x, u)
            X.append(x)  # updated states trajectories
            Reward.append(reward)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        return np.asarray(X), np.asarray(Reward)



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

    # building model
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
    plot.pltPhase(X=Y)

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
    # simulate open loop
    X = cstr_model.simulate(ninit, nsim, ts, U)
    # plot trajectories
    plot.pltOL(Y=X[:,0], U=U, X=X[:,1])
    plot.pltPhase(X=X)

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
    # simulate open loop
    X = twotank_model.simulate(ninit, nsim, ts, pump, valve)
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
    tank_model = Tank()  # instantiate CSTR class
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
    seir_model = SEIR_population()  # instantiate CSTR class
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
    lcp_model = LinCartPole()  # instantiate CSTR class
    lcp_model.parameters()
    # simulate open loop
    X, Y = lcp_model.simulate(ninit=ninit, nsim=nsim, ts=ts, U=U)
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
    gym_model.parameters(env_pick=environment)
    ninit = 0
    nsim = 201
    U = np.zeros([(nsim - 1), gym_model.nu])
    if type(gym_model.action_sample) == int:
        U = U.astype(int)
    X, Reward = gym_model.simulate(ninit=ninit, nsim=nsim, U=U)
    plot.pltOL(Y=Reward, X=X, U=U)
    plot.pltPhase(X=X)
    # TODO: include visualization option for the trajectories or render of OpenAI gym


    # Lorenz 96
    ninit = 0
    nsim = 3001
    ts = 0.01
    #  inverted pendulum
    lorenz96_model = Lorenz96()  # instantiate CSTR class
    lorenz96_model.parameters()
    # simulate open loop
    X = lorenz96_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # LorenzSystem
    ninit = 0
    nsim = 4001
    ts = 0.01
    #  inverted pendulum
    lorenz_model = LorenzSystem()  # instantiate CSTR class
    lorenz_model.parameters()
    # simulate open loop
    X = lorenz_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    # phase plot
    plot.pltPhase(X=X)

    # VanDerPol
    ninit = 0
    nsim = 401
    ts = 0.1
    #  inverted pendulum
    vdp_model = VanDerPol()  # instantiate CSTR class
    vdp_model.parameters()
    # simulate open loop
    X = vdp_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # HindmarshRose
    ninit = 0
    nsim = 5001
    ts = 0.1
    U = 3*np.asarray([np.ones((nsim - 1))]).T
    HR_model = HindmarshRose()  # instantiate CSTR class
    HR_model.parameters()
    # simulate open loop
    X = HR_model.simulate(ninit=ninit, nsim=nsim, U=U, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X,U=U)
    plot.pltPhase(X=X)

    # ThomasCSattractor
    ninit = 0
    nsim = 5001
    ts = 0.1
    TCSA_model = ThomasAttractor()  # instantiate
    TCSA_model.parameters()
    # simulate open loop
    X = TCSA_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # RösslerAttractor
    ninit = 0
    nsim = 20001
    ts = 0.01
    Ross_model = RosslerAttractor()  # instantiate
    Ross_model.parameters()
    # simulate open loop
    X = Ross_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # LotkaVolterra
    ninit = 0
    nsim = 1001
    ts = 0.1
    #  inverted pendulum
    lv_model = LotkaVolterra()  # instantiate CSTR class
    lv_model.parameters()
    # simulate open loop
    X = lv_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # Brusselator1D
    ninit = 0
    nsim =  501
    ts = 0.1
    #  inverted pendulum
    bruss_model = Brusselator1D()  # instantiate CSTR class
    bruss_model.parameters()
    # simulate open loop
    X = bruss_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

    # ChuaCircuit
    ninit = 0
    nsim = 10001
    ts = 0.01
    #  inverted pendulum
    chua_model = ChuaCircuit()  # instantiate CSTR class
    chua_model.parameters()
    # simulate open loop
    X = chua_model.simulate(ninit=ninit, nsim=nsim, ts=ts)
    # plot trajectories
    plot.pltOL(Y=X)
    plot.pltPhase(X=X)

