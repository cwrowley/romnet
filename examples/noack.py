#!/usr/bin/env python

import numpy as np
import romnet


class NoackModel(romnet.Model):
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


def random_ic():
    xmax = 4
    zmin = -4
    zmax = 4
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    z = zmin + (zmax - zmin) * np.random.rand()
    return np.array((x, y, z))


def generate_data():
    model = NoackModel(mu=0.1, omega=2., A=-0.1, lam=10)
    dt = 0.1
    model.set_stepper(dt, method="rk2", nsteps=5)

    # generate trajectories for training/testing
    num_train = 1024
    num_test = 64
    n = 30  # length of each trajectory
    print("Generating training trajectories...")
    training_traj = romnet.sample(model.step, random_ic, num_train, n)
    test_traj = romnet.sample(model.step, random_ic, num_test, n)

    # sample gradients for GAP loss
    s = 32  # samples per trajectory
    L = 15  # horizon for gradient sampling
    print("Sampling gradients...")
    training_data = romnet.sample_gradient(training_traj, model, s, L)
    test_data = romnet.sample_gradient(test_traj, model, s, L)
    print("Done")

    training_data.save("noack_train.dat")
    test_data.save("noack_test.dat")


if __name__ == "__main__":
    generate_data()
