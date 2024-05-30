"""
Nonlinear ODEs. Wrapper for emulator dynamical models

    * Internal Emulators - in house ground truth equations
    * External Emulators - third party models

References:

    * https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
    * https://en.wikipedia.org/wiki/List_of_dynamical_systems_and_differential_equations_topics
"""
import inspect, sys
from neuromancer.psl.base import ODE_Autonomous, cast_backend


class UniversalOscillator(ODE_Autonomous):
    """
    Harmonic oscillator

    * https://en.wikipedia.org/wiki/Harmonic_oscillator
    * https://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    """
    @property
    def params(self):
        variables = {'x0': [1.0, 0.0]}
        constants = {'ts': 0.1}
        parameters = {'mu': 2., 'omega': 1.}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = x[1]
        dx2 = -2.*self.mu*x[1] - x[0] + self.B.core.cos(self.omega*t)
        return [dx1, dx2]


class Pendulum(ODE_Autonomous):
    """
    `Simple pendulum <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html>`_
    """
    @property
    def params(self):
        variables = {'x0': [0., 1.],}
        constants = {'ts': 0.1}
        parameters = {'g': 9.81, 'f': 3,}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        theta = x[0]
        omega = x[1]
        return [omega, -self.f*omega - self.g*self.B.core.sin(theta)]


class DoublePendulum(ODE_Autonomous):
    """
    `Double Pendulum <https://scipython.com/blog/the-double-pendulum/>`_
    """
    @property
    def params(self):
        variables = {'x0': [3. * self.B.core.pi / 7., 0., 3. * self.B.core.pi / 4., 0.],}
        constants = {'ts': 0.1}
        parameters = {'L1': 1.,
                      'L2': 1.,
                      'm1': 1.,
                      'm2': 1.,
                      'g': 9.81,
                      }
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        theta1 = x[0]
        z1 = x[1]
        theta2 = x[2]
        z2 = x[3]
        c, s = self.B.core.cos(theta1 - theta2), self.B.core.sin(theta1 - theta2)
        dx1 = z1
        dx2 = (self.m2 * self.g * self.B.core.sin(theta2) * c - self.m2 * s * (self.L1 * z1 ** 2 * c + self.L2 * z2 ** 2) -
               (self.m1 + self.m2) * self.g * self.B.core.sin(theta1)) / self.L1 / (self.m1 + self.m2 * s ** 2)
        dx3 = z2
        dx4 = ((self.m1 + self.m2) * (self.L1 * z1 ** 2 * s - self.g * self.B.core.sin(theta2) + self.g * self.B.core.sin(theta1) * c) +
               self.m2 * self.L2 * z2 ** 2 * s * c) / self.L2 / (self.m1 + self.m2 * s ** 2)
        return [dx1, dx2, dx3, dx4]


class LorenzSystem(ODE_Autonomous):
    """
    Lorenz System

    * https://en.wikipedia.org/wiki/Lorenz_system#Analysis
    * https://ipywidgets.readthedocs.io/en/stable/examples/Lorenz%20Differential%20Equations.html
    * https://scipython.com/blog/the-lorenz-attractor/
    * https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    """
    @property
    def params(self):
        variables = {'x0': [1.0, 1.0, 1.0]}
        constants = {'ts': 0.1}
        parameters = {'sigma': 10.,
                      'beta': 8./3.,
                      'rho': 28.0}
        meta = {}
        return variables, constants, parameters, meta


    @cast_backend
    def equations(self, t, x):
        dx1 = self.sigma*(x[1] - x[0])
        dx2 = x[0]*(self.rho - x[2]) - x[1]
        dx3 = x[0]*x[1] - self.beta*x[2]
        return [dx1, dx2, dx3]


class VanDerPol(ODE_Autonomous):
    """
    Van der Pol oscillator

    * https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    * http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    """
    @property
    def params(self):
        variables = {'x0': [1., 2.]}
        constants = {'ts': 0.1}
        parameters = {'mu': 1.}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = self.mu*(x[0] - 1./3.*x[0]**3 - x[1])
        dx2= x[0]/self.mu
        return [dx1, dx2]


class ThomasAttractor(ODE_Autonomous):
    """
    Thomas' cyclically symmetric attractor

    * https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor
    """
    @property
    def params(self):
        variables = {'x0': [1., -1., 1.]}
        constants = {'ts': 0.1}
        parameters = {'b': 0.208186}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = self.B.core.sin(x[1]) - self.b*x[0]
        dx2 = self.B.core.sin(x[2]) - self.b*x[1]
        dx3 = self.B.core.sin(x[0]) - self.b*x[2]
        return [dx1, dx2, dx3]


class RosslerAttractor(ODE_Autonomous):
    """
    `Rössler attractor <https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor>`_
    """
    @property
    def params(self):
        variables = {'x0': [0., 0., 0.]}
        constants = {'ts': 0.1}
        parameters = {'a': 0.2,
                      'b': 0.2,
                      'c': 5.7,}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = - x[1] - x[2]
        dx2 = x[0] + self.a*x[1]
        dx3 = self.b + x[2]*(x[0]-self.c)
        return [dx1, dx2, dx3]


class LotkaVolterra(ODE_Autonomous):
    """
    `Lotka–Volterra equations <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
    Also known as the predator–prey equations
    """
    @property
    def params(self):
        variables = {'x0': [5., 100.]}
        constants = {'ts': 0.1}
        parameters = {'a': 1.1,
                      'b': 0.4,
                      'c': 0.1,
                      'd': 0.4,}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = self.a*x[0] - self.b*x[0]*x[1]
        dx2 = self.c**x[0]*x[1] - self.d*x[1]
        return [dx1, dx2]


class Brusselator1D(ODE_Autonomous):
    """
    `Brusselator <https://en.wikipedia.org/wiki/Brusselator>`_
    """
    @property
    def params(self):
        variables = {'x0': [1., 1.]}
        constants = {'ts': 0.1}
        parameters = {'a': 1.,
                      'b': 3.}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = self.a + x[1]*x[0]**2 -self.b*x[0] - x[0]
        dx2 = self.b*x[0] - x[1]*x[0]**2
        return [dx1, dx2]


class ChuaCircuit(ODE_Autonomous):
    """
    Chua's circuit

    * https://en.wikipedia.org/wiki/Chua%27s_circuit
    * https://www.chuacircuits.com/matlabsim.php
    """
    @property
    def params(self):
        variables = {'x0': [0.7, 0.0, 0.0],}
        constants = {'ts': 0.1}
        parameters = {'a': 15.6,
                      'b': 28.,
                      'm0': -1.143,
                      'm1': -0.714,}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        fx = self.m1*x[0] + 0.5*(self.m0 - self.m1)*(self.B.core.abs(x[0] + 1) - self.B.core.abs(x[0] - 1))
        dx1 = self.a*(x[1] - x[0] - fx)
        dx2 = x[0] - x[1] + x[2]
        dx3 = -self.b*x[1]
        return [dx1, dx2, dx3]


class Duffing(ODE_Autonomous):
    """
    `Duffing equation <https://en.wikipedia.org/wiki/Duffing_equation>`_
    """
    @property
    def params(self):
        variables = {'x0': [1.0, 0.0]}
        constants = {'ts': 0.01}
        parameters = {'alpha': 1.,
                      'beta': 5.,
                      'delta': 0.02,
                      'gamma': 8.,
                      'omega': 0.5,}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        dx1 = x[1]
        dx2 = - self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3 + self.gamma*self.B.core.cos(self.omega*t)
        return [dx1, dx2]


class Autoignition(ODE_Autonomous):
    """
    ODE describing pulsating instability in open-ended combustor.

    * Koch, J., Kurosaka, M., Knowlen, C., Kutz, J.N.,
      "Multiscale physics of rotating detonation waves: Autosolitons and modulational instabilities,"
      Physical Review E, 2021
    """
    @property
    def params(self):
        variables = {'x0': [1., 0.7],}
        constants = {'ts': 0.1}
        parameters = {'alpha': 0.3,
                      'uc': 1.1,
                      's': 1.,
                      'k': 1.,
                      'r': 5.,
                      'q': 6.5,
                      'up': 0.55,
                      'e': 1.}
        meta = {}
        return variables, constants, parameters, meta

    @cast_backend
    def equations(self, t, x):
        reactionRate = self.k * (1.0 - x[1]) * self.B.core.exp((x[0] - self.uc) / self.alpha)
        regenRate = self.s * self.up * x[1] / (1.0 + self.B.core.exp(self.r * (x[0] - self.up)))
        dx1 = self.q * reactionRate - self.e * x[0] ** 2
        dx2 = reactionRate - regenRate
        return [dx1, dx2]


systems = dict(inspect.getmembers(sys.modules[__name__], lambda x: inspect.isclass(x)))
systems = {k: v for k, v in systems.items() if issubclass(v, ODE_Autonomous) and v is not ODE_Autonomous}

if __name__ == '__main__':
    for n, system in systems.items():
        print(n)
        s = system()
        out = s.simulate(nsim=5)
        print({k: v.shape for k, v in out.items()})
