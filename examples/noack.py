#!/usr/bin/env python

import numpy as np

import romnet
from romnet.models import NoackModel


def random_ic():
    # Old code used these values for test trajectory visualization and error  -
    # calculation
    # xmax = 1.5
    # zmin = -1.5
    # zmax = 1.5
    xmax = 6
    zmin = -6
    zmax = 6
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    z = zmin + (zmax - zmin) * np.random.rand()
    return np.array((x, y, z))


def generate_data():
    model = NoackModel(mu=0.1, omega=2., A=-0.1, lam=10)
    dt = 0.1
    model.set_stepper(dt, method="rk2", nsteps=5)

    # generate trajectories for training/testing
    num_train = 1000
    num_test = 64  # Old code used 1000                                        -
    n = 200  # length of each trajectory
    print("Generating training trajectories...")
    training_traj = romnet.sample(model.step, random_ic, num_train, n)
    test_traj = romnet.sample(model.step, random_ic, num_test, n)
    training_traj.save("noack_train.traj")
    test_traj.save("noack_test.traj")

    # sample gradients for GAP loss
    s = 10  # samples per trajectory
    L = 20  # horizon for gradient sampling
    print("Sampling gradients...")
    training_data, _ = romnet.sample_gradient_long_traj(training_traj, model, s, L)
    test_data, _ = romnet.sample_gradient_long_traj(test_traj, model, s, L)
    print("Done")

    training_data.save("noack_train.dat")
    test_data.save("noack_test.dat")


if __name__ == "__main__":
    generate_data()
