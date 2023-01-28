"""Model of cylinder wake, used in Noack (2000)"""

import numpy as np

from .. import Model

__all__ = ["NoackModel"]


class NoackModel(Model):
    """
    3-state ODE model, normal form of Hopf bifurcation, used in Noack (2000)
    """
    state_dim = 3
    output_dim = 3

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
