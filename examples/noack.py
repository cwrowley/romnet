#!/usr/bin/env python

import numpy as np

import romnet
from romnet.models import NoackModel


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
