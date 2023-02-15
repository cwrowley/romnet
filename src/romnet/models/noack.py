"""Model of cylinder wake, used in Noack (2000)"""

import numpy as np

from .. import Model

__all__ = ["NoackModel"]


class NoackModel(Model):
    """
    3-state ODE model, normal form of Hopf bifurcation, used in Noack (2000)
    """
    num_states = 3
    num_outputs = 3

    def __init__(self, mu=0.1, omega=1., A=-0.1, lam=10.):
        self.mu = mu
        self.omega = omega
        self.A = A
        self.lam = lam

    def rhs(self, x):
        xdot = self.mu * x[0] - self.omega * x[1] + self.A * x[0] * x[2]
        ydot = self.omega * x[0] + self.mu * x[1] + self.A * x[1] * x[2]
        zdot = -self.lam * (x[2] - x[0]**2 - x[1]**2)
        return np.array([xdot, ydot, zdot])

    def jac(self, x):
        df1 = [self.mu + self.A * x[2], -self.omega, self.A * x[0]]
        df2 = [self.omega, self.mu + self.A * x[2], self.A * x[1]]
        df3 = [2 * self.lam * x[0], 2 * self.lam * x[1], -self.lam]
        return np.array([df1, df2, df3])

    def adjoint_rhs(self, x, v):
        return self.jac(x).T @ v

    def slow_manifold(self, r):
        h0 = r**2
        h1 = - 2 * r**2 * (self.A * r**2 + self.mu)
        h2 = 4 * r**2 * (3 * self.A**2 * r**4 + 4 * self.A * r**2 * self.mu +
                         self.mu**2)
        h3 = -8 * r**2 * (14 * self.A**3 * r**6 + 24 * self.A**2 * r**4 *
                          self.mu + 11 * self.A * r**2 * self.mu**2 +
                          self.mu**3)
        h4 = 16 * r**2 * (85 * self.A**4 * r**8 + 180 * self.A**3 *
                          r**6 * self.mu + 120 * self.A**2 * r**4 *
                          self.mu**2 + 26 * self.A * r**2 * self.mu**3
                          + self.mu**4)
        return (h0 + h1 * (1/self.lam) + h2 * (1/self.lam)**2 +
                h3 * (1/self.lam)**3 + h4 * (1/self.lam)**4)
