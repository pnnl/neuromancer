"""
Non-autonomous dynamic systems.
`Reference for Chaotic nonlinear ODEs <https://en.wikipedia.org/wiki/List_of_chaotic_maps>`_
"""
import numpy as np
from neuromancer.psl.signals import step, sines, periodic, noise, walk
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl.base import cast_backend
import inspect, sys


class LorenzControl(ODE):

    @property
    def params(self):
        variables = {'x0': [-8., 8., 27.]}
        constants = {'ts': 0.01}
        parameters = {'sigma': 10.,
                      'beta': 8. / 3.,
                      'rho': 28.0}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        t = self.rng.uniform(low=0, high=np.pi)
        self.ninit = t
        T = self.B.core.arange(t, t + self.ts * nsim, self.ts)
        u = self.u_fun(T).T[:nsim]
        return u

    @cast_backend
    def equations(self, t, x, u):

        return self.B.cast([self.sigma * (x[1] - x[0]) + u[0],
                            x[0] * (self.rho - x[2]) - x[1],
                            x[0] * x[1] - self.beta * x[2] - u[1]])

    def u_fun(self, t):
        u = self.B.core.stack([self.B.core.sin(2 * t), self.B.core.sin(8 * t)])
        return u


class SEIR_population(ODE):
    """
    Susceptible, Exposed, Infected, and Recovered (SEIR) population population model.
    Used to model COVID-19 spread.
    `Source of the model <https://apmonitor.com/do/index.php/Main/COVID-19Response>`_

    states:

    * Susceptible (s): population fraction that is susceptible to the virus
    * Exposed (e): population fraction is infected with the virus but does not transmit to others
    * Infectious (i): population fraction that is infected and can infect others
    * Recovered (r): population fraction recovered from infection and is immune from further infection
    """
    @property
    def params(self):
        variables = {'x0': [1 - 1./10000. - 0. - 0., 1./10000., 0., 0.],}  # [s0, e0, i0, r0]
        constants = {'ts': 0.01,
                     'N': 10000}
        parameters = {'e_0': 1./10000.,  # 1/N
                      'i_0': 0.,
                      'r_0': 0.,
                      's_0': 1 - 1./10000. - 0. - 0.,  # 1 - e0 - i0 - r0
                      't_incubation': 5.1,
                      't_infective': 3.3,
                      'R0': 2.4,
                      'alpha': 1./5.1,  # 1/t_incubation
                      'gamma': 1./3.3,  # 1 / t_infective
                      'beta': 2.4 * (1./3.3),  # R0 * gamma
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        return step(nsim=nsim, d=2, randsteps=int(np.ceil(nsim / 24)), min=0., max=1., rng=self.rng)

    @cast_backend
    def equations(self, t, x, u):
        """

        *  Inputs (1): social distancing (u=0 (none), u=1 (total isolation))
        *  States (4):
            *  Susceptible (s): population fraction that is susceptible to the virus
            *  Exposed (e): population fraction is infected with the virus but does not transmit to others
            *  Infectious (i): population fraction that is infected and can infect others
            *  Recovered (r): population fraction recovered from infection and is immune from further infection
        """

        s = x[0]
        e = x[1]
        i = x[2]
        u = u[0]

        sdt = -(1 - u) * self.beta * s * i
        edt = (1 - u) * self.beta * s * i - self.alpha * e
        idt = self.alpha * e - self.gamma * i
        rdt = self.gamma * i
        dx = [sdt, edt, idt, rdt]
        return dx


class Tank(ODE):
    """
    Single Tank model
    `Original code obtained from APMonitor <https://apmonitor.com/pdc/index.php/Main/TankLevel>`_
    """
    @property
    def params(self):
        variables = {'x0': np.array([0.]),}
        constants = {'ts': 0.1,}
        parameters = {'rho': 1000.,  # water density (kg/m^3)
                      'A': 1.,  # tank area (m^2)
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_x0(self):
        return self.rng.uniform(low=0.0, high=0.1, size=(1,))

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        c = step(nsim=nsim, d=1, min=0.1, max=55., randsteps=int(np.ceil(nsim / 48)), rng=self.rng)
        valve = periodic(nsim=nsim, d=1, min=0., max=10., periods=int(np.ceil(nsim / 48)), form='sin', rng=self.rng) + noise(nsim=nsim, d=1, rng=self.rng)
        return np.concatenate([c, valve], axis=1)

    @cast_backend
    def equations(self, t, x, u):
        """
        * States (1): level in the tanks
        * Inputs u(1): c - valve coefficient (kg/s / %open)
        * Inputs u(2): valve in % [0-100]
        """

        c = u[0]
        valve = u[1]
        dx_dt = (c / (self.rho*self.A)) * valve
        return dx_dt


class TwoTank(ODE):
    """
    Two Tank model.
    `Original code obtained from APMonitor <https://apmonitor.com/do/index.php/Main/LevelControl>`_
    """
    @property
    def params(self):
        variables = {'x0': [0., 0.],}
        constants = {'ts': 1.0}
        parameters = {'c1': 0.08,  # inlet valve coefficient
                      'c2': 0.04,  # tank outlet coefficient
                      }
        meta = {}
        return variables, constants, parameters, meta

    @property
    def umin(self):
        return np.array([0.0, 0.0], dtype=np.float32)

    @property
    def umax(self):
        """
        Note that although the theoretical upper bound is 1.0,
        this results in numerical instability in the integration.
        """
        return np.array([0.5, 0.5], dtype=np.float32)

    @cast_backend
    def get_x0(self):
        return self.rng.uniform(low=0.0, high=0.5, size=2)

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = step(nsim=nsim, d=2, min=0., max=0.4,
                 randsteps=int(np.ceil(self.ts*nsim/100)), rng=self.rng)
        return u

    @cast_backend
    def equations(self, t, x, u):

        h1 = self.B.core.clip(x[0], 0, 1)  # States (2): level in the tanks
        h2 = self.B.core.clip(x[1], 0, 1)
        pump = self.B.core.clip(u[0], 0, 1)  # Inputs (2): pump and valve
        valve = self.B.core.clip(u[1], 0, 1)
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * self.B.core.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * self.B.core.sqrt(h1) - self.c2 * self.B.core.sqrt(h2)
        if h1 >= 1.0 and dhdt1 > 0.0:
            dhdt1 = 0
        if h2 >= 1.0 and dhdt2 > 0.0:
            dhdt2 = 0
        dhdt = [dhdt1, dhdt2]
        return dhdt


class CSTR(ODE):
    """
    Continuous Stirred Tank Reactor model
    `Original code obtained from APMonitor <http://apmonitor.com/do/index.php/Main/NonlinearControl>`_
    """
    @property
    def params(self):
        variables = {'x0': [0.87725294608097, 324.475443431599],}  # [Ca, T] Steady State Initial Condition for the Uncontrolled Inputs}
        constants = {'ts': 0.1,
                     'Ca_ss': 0.87725294608097,
                     'T_ss': 324.475443431599,}
        parameters = {'q': 100.,  # Volumetric Flowrate (m^3/sec)
                      'V': 100.,  # Volume of CSTR (m^3)
                      'rho': 1000.,  # Density of A-B Mixture (kg/m^3)
                      'Cp': 0.239,  # Heat capacity of A-B Mixture (J/kg-K)
                      'mdelH': 5.0e4,  # Heat of reaction for A->B (J/mol)
                      'EoverR': 8750., # E - Activation energy in the Arrhenius Equation (J/mol),  R - Universal Gas Constant = 8.31451 J/mol-K
                      'k0': 7.2e10,  # Pre-exponential factor (1/sec)
                      'UA': 5.0e4,  # U - Overall Heat Transfer Coefficient (W/m^2-K),  A - Area - this value is specific for the U calculation (m^2)
                      'u_ss': 300.0,  # cooling jacket Temperature (K)
                      'Tf': 350.,  # Feed Temperature (K)
                      'Caf': 1.,  # Feed Concentration (mol/m^3),
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_x0(self, rand=False):
        if rand:
            x1 = self.rng.normal(loc=self.Ca_ss)
            x2 = self.rng.normal(loc=self.T_ss, scale=0.01)
            return [x1, x2]
        else:
            return [self.Ca_ss, self.T_ss]

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        U = 300. + step(nsim=nsim, d=1, min=-3., max=3., randsteps=int(np.ceil(nsim / 25)), rng=self.rng)
        return U

    @cast_backend
    def equations(self, t, x, u):
        """
        Inputs (1):
            * Temperature of cooling jacket (K)
        Disturbances (2):
            * Tf = Feed Temperature (K)
            * Caf = Feed Concentration (mol/m^3)
        States (2):
            * Concentration of A in CSTR (mol/m^3)
            * Temperature in CSTR (K)
        """
        Tc = u  # Temperature of cooling jacket (K)
        Ca = x[0]  # Concentration of A in CSTR (mol/m^3)
        T = x[1]  # Temperature in CSTR (K)
        rA = self.k0 * self.B.core.exp(-self.EoverR / T) * Ca  # reaction rate
        dCadt = self.q / self.V * (self.Caf - Ca) - rA  # Calculate concentration derivative
        dTdt = self.q / self.V * (self.Tf - T) \
               + self.mdelH / (self.rho * self.Cp) * rA \
               + self.UA / self.V / self.rho / self.Cp * (Tc - T)  # Calculate temperature derivative
        return [dCadt, dTdt]


class InvPendulum(ODE):
    """
    Inverted Pendulum dynamics
    * states: :math:`x = [\theta \dot{\theta}]`;
        * :math:`\theta` is angle from upright equilibrium
    * input: u = input torque

    """
    @property
    def params(self):
        variables = {'x0': [0.5, 0.]}
        constants = {'ts': 0.1}
        parameters = {'g': 9.81,  # Acceleration due to gravity (m/s^2)
                      'L': 0.5,  # Length of pole in m
                      'm': 0.15,  # ball mass in kg
                      'b': 0.1,  # friction
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        return 0.1*self.rng.normal(size=(nsim, 1))

    @cast_backend
    def equations(self, t, x, u):
        y = [x[1],
             (self.m * self.g * self.L * np.sin(x[0]) - self.b * x[1]) / (self.m * self.L ** 2)]
        y[1] = y[1] + (u / (self.m * self.L ** 2))
        return y


class HindmarshRose(ODE):
    """
    Hindmarsh–Rose model of neuronal activity

    * https://en.wikipedia.org/wiki/Hindmarsh%E2%80%93Rose_model
    * https://demonstrations.wolfram.com/HindmarshRoseNeuronModel/
    """
    @property
    def params(self):
        variables = {'x0': [-5., -10., 0.]}
        constants = {'ts': 0.1}
        parameters = {'a': 1.,
                      'b': 2.6,
                      'c': 1.,
                      'd': 5.,
                      's': 4.,
                      'xR': -8./5.,
                      'r': 0.01,
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        return step(nsim=nsim, d=1, min=2.99, max=3.1, randsteps=int(max(1, nsim/48)), rng=self.rng)

    @cast_backend
    def equations(self, t, x, u):
        theta = -self.a*x[0]**3 + self.b*x[0]**2
        phi = self.c -self.d*x[0]**2
        dx1 = x[1] + theta - x[2] + u
        dx2 = phi - x[1]
        dx3 = self.r*(self.s*(x[0]-self.xR)-x[2])
        dx = [dx1, dx2, dx3]
        return dx


class IverSimple(ODE):
    """
    Dynamic model of Unmanned Underwater Vehicle (modified from Stankiewicz et al)
    -- Excludes rolling, sway, currents, Includes: hydrostate/dynamic terms,
    control surface deflections/propeller thrust, and actuator dynamics
    with non-kinematic output
    """
    @property
    def params(self):
        variables = {'x0': [0., 0., 0.01, 0., 0., 0., 0., 0.]}
        constants = {'ts': 0.01}
        parameters = {'Mq': -0.748,  # Hydrodynamic coefficient (1/s)
                      'Nur': -0.441,  # Hydrodynamic coefficient (1/m)
                      'Xuu': -0.179,  # Hydrodynamic coefficient (1/m)
                      'Zww': 0.098,  # Hydrodynamic coefficient (1/m)
                      'Muq': -3.519,  # Hydrodynamic coefficient (1/m)
                      'Bz': 8.947,  # Bouyancy term that accounts for the center of bouyancy vertical offset from the center of gravity (1/s^2)
                      'k': 0.519,  # Hydrodynamic coefficient (m/s^2)
                      'b': 3.096,  # Hydrodynamic coefficient (1/m^2)
                      'c': 0.065,  # Hydrodynamic coefficient (1/m^2)
                      'K_delta_u': -10.0,  # Thruster dynamic coefficient
                      'K_delta_q': -10.0,  # Elevator deflection dynamic coefficient
                      'K_delta_r': -10.0,  # Rudder deflection dynamic coefficient
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        delta = step(nsim=nsim, d=3, min=[0.0, -1.0, -1.0], max=[1.0, 1., 1.], randsteps=100, rng=self.rng)
        return delta

    @cast_backend
    def equations(self, t, x, u):
        """
        * States (8): [theta, psi, uu, q, r, delta_u, delta_q, delta_r]
        * Inputs (3): [delta_uc, delta_qc, delta_rc] (thrust speed/deflections, normalized)
        """
        theta = x[0]  # States
        psi = x[1]
        uu = x[2]
        q = x[3]
        r = x[4]
        delta_u = x[5]
        delta_q = x[6]
        delta_r = x[7]

        delta_uc = u[0]  # Control
        delta_qc = u[1]
        delta_rc = u[2]

        dx_dt = np.zeros(8)  # Kinematics:
        dx_dt[0] = q
        dx_dt[1] = r / (np.cos(theta))

        dx_dt[2] = self.Xuu*(uu**2) + self.k*delta_u  # Dynamics
        dx_dt[3] = self.Muq*uu*q + self.Mq*q - self.Bz*np.sin(theta) + self.b*(uu**2)*delta_q
        dx_dt[4] = self.Nur*uu*r + self.c*(uu**2)*delta_r

        dx_dt[5] = self.K_delta_u*( delta_u - delta_uc )  # Actuator dynamics
        dx_dt[6] = self.K_delta_q*( delta_q - delta_qc )
        dx_dt[7] = self.K_delta_r*( delta_r - delta_rc )

        return dx_dt


class Actuator(ODE):
    """
    These are the actuator dynamics from the IVER systems.
    Since the equations are linear they are a good sanity check for your modeling implementations.
    """

    @property
    def params(self):
        variables = {'x0': [0., 0., 0.]}
        constants = {'ts': 0.1}
        parameters = {'K_delta_u': -10.0,  # Thruster dynamic coefficient
                      'K_delta_q': -10.0,  # Elevator deflection dynamic coefficient
                      'K_delta_r': -10.0   # Rudder deflection dynamic coefficient
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        delta = sines(nsim=nsim, d=3, min=[0.0, -1.0, -1.0], max=[1.0, 1., 1.], periods=100, rng=self.rng)
        return delta

    @cast_backend
    def equations(self, t, x, u):
        """
        * States (8): [delta_u, delta_q, delta_r]
        * Inputs (3): [delta_uc, delta_qc, delta_rc] (thrust speed/deflections, normalized)
        """
        # States
        delta_u = x[0]
        delta_q = x[1]
        delta_r = x[2]

        # Control
        delta_uc = u[0]
        delta_qc = u[1]
        delta_rc = u[2]

        # Actuator dynamics:
        dx_dt = np.zeros(3)
        dx_dt[0] = self.K_delta_u * (delta_u - delta_uc)
        dx_dt[1] = self.K_delta_q * (delta_q - delta_qc)
        dx_dt[2] = self.K_delta_r * (delta_r - delta_rc)

        return dx_dt


class SwingEquation(ODE):
    """
    `Power Grid Swing Equation. <https://en.wikipedia.org/wiki/Swing_equation>`_
    The second-order swing equation is converted to two first-order ODEs
    """
    @property
    def params(self):
        Pm = 0.8
        Pmax = 5.0
        H = 500.
        freq = 60.
        ws = 2 * np.pi * freq
        variables = {'x0': [np.arcsin(Pm/Pmax), 0.]}
        constants = {'ts': 0.01}
        parameters = {'Pm': Pm,  # Mechanical power
                      'Pmax': Pmax,  # Maximum electrical output
                      'H': H,  # Inertia constant
                      'D': 5.,  # Damping coefficient
                      'freq': freq,  # Base frequency
                      'ws': ws,  # Base angular speed
                      'M': 2 * H / ws  # scaled inertia constant
                       }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        """
        Noisy mechanical power with constant Pmax)
        """
        u = step(nsim=nsim, d=1,  min=0.8 * 0.98, max=0.8 * 1.02,
                 randsteps=int(np.ceil(nsim / 200)),
                 rng=self.rng)
        return u

    @cast_backend
    def equations(self, t, x, u):
        delta = x[0]
        domega = x[1]
        Pm = u[0]
        Pmax = self.Pmax
        dx_dt = [self.ws * domega,
                 (Pm - Pmax * np.sin(delta) - self.D * domega) / self.M]
        return dx_dt


class DuffingControl(ODE):
    """
    Duffing equation with driving force as a function of control inputs not time
    `Source <https://en.wikipedia.org/wiki/Duffing_equation>`_
    """
    @property
    def params(self):
        variables = {'x0': [1., 0.],
                     'U': 0.5*np.cos([np.arange(0., self.nsim+1) * 0.1]).T}
        constants = {'ts': 0.1}
        parameters = {'alpha': 1.,
                      'delta': 0.02,
                      'beta': 5.,
                      'gamma': 8.,
                      'omega': 0.5,
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = periodic(nsim, d=1, min=0., max=5.,
                     periods=int(np.ceil(nsim/100)),
                     form='sin')

        return u

    @cast_backend
    def equations(self, t, x, u):
        dx1 = x[1]
        dx2 = - self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3 + \
              self.gamma*np.cos(self.omega*u[0])
        dx = [dx1, dx2]
        return dx


class VanDerPolControl(ODE):
    """
    Van der Pol oscillator

    * https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    * http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    * section V.A in: https://arxiv.org/abs/2203.14114
    """
    @property
    def params(self):
        variables = {'x0': self.rng.standard_normal(2),
                     'U': 0.5*np.cos([np.arange(0., self.nsim+1) * 0.02]).T}
        constants = {'ts': 0.1}
        parameters = {'mu': 1.0}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x, u):
        dx1 = x[1]
        dx2 = self.mu*(1 - x[0]**2)*x[1] - x[0] + u[0]
        dx = [dx1, dx2]
        return dx

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = step(nsim=nsim, d=1,  min=-5., max=5.,
                 randsteps=int(np.ceil(nsim / 200)),
                 rng=self.rng)
        return u


class ThomasAttractorControl(ODE):
    """
    Thomas' cyclically symmetric attractor
    control input: dissipativity parameter b
    `Source <https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor>`_
    """
    @property
    def params(self):
        variables = {'x0': [1., -1., 1.]}
        constants = {'ts': 0.1}
        parameters = {'b': 0.208186}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        return step(nsim=nsim, d=1, values=[0.208186, 0.3289, 0.5, 1.0], min=0., max=1.,
                    randsteps=int(np.ceil(nsim / 48)), rng=self.rng) + noise(nsim=nsim, d=1, min=-.01, max=.01, rng=self.rng)

    @cast_backend
    def equations(self, t, x, u):
        b = u[0]
        dx1 = np.sin(x[1]) - b*x[0]
        dx2 = np.sin(x[2]) - b*x[1]
        dx3 = np.sin(x[0]) - b*x[2]
        dx = [dx1, dx2, dx3]
        return dx


systems = dict(inspect.getmembers(sys.modules[__name__], lambda x: inspect.isclass(x)))
systems = {k: v for k, v in systems.items() if issubclass(v, ODE) and v is not ODE}

if __name__ == '__main__':
    for n, system in systems.items():
        print(n)
        s = system()
        out = s.simulate(nsim=5)
        print({k: v.shape for k, v in out.items()})

