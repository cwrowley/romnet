import numpy as np
from romnet import Model

class SlowSubspace(Model):
    """
    3-state ODE model with a slow subspace and transient growth
    """
    state_dim = 3
    output_dim = 3

    def __init__(self, mu=0.1, omega=0.1, phi=np.pi/4, lam=20.):
        self.mu = mu
        self.omega = omega
        self.phifac = 1. / np.tan(phi)
        self.lam = lam

    def rhs(self, x):
        xdot = (self.mu * x[0] - self.omega * x[1] -  self.mu * x[0] *
                (x[0]**2 + x[1]**2))
        ydot = (self.omega * x[0] + self.mu * x[1] - self.mu * x[1] *
                (x[0]**2 + x[1]**2) - self.lam * x[2] * self.phifac)
        zdot = -self.lam * x[2]
        return np.array([xdot, ydot, zdot])

    def jac(self, x):
        df1 = [self.mu * (1 - 2 * x[0]**2 - x[1]**2),
               -self.omega - 2 * self.mu * x[0] * x[1],
               0]
        df2 = [self.omega - 2 * self.mu * x[0] * x[1],
               self.mu * (1 - x[0]**2 - 3 * x[1]**2),
               -self.lam * self.phifac]
        df3 = [0, 0, -self.lam]
        return np.array([df1, df2, df3])

    def adjoint_rhs(self, x, v):
        return self.jac(x).T @ v

def random_ic():
    xmax = 4
    zmin = -4
    zmax = 4
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    z = zmin + (zmax - zmin) * np.random.rand()
    return np.array((x,y,z))
