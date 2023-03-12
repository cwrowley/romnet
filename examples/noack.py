#!/usr/bin/env python

import numpy as np

import romnet
from romnet.models import NoackModel
from romnet.typing import Vector


def random_ic() -> Vector:
    xmax = 6
    zmin = -6
    zmax = 6
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    z = zmin + (zmax - zmin) * np.random.rand()
    return np.array((x, y, z))


def identity(x: Vector) -> Vector:
    return x


def adj_output(x: Vector, eta: Vector) -> Vector:
    return eta


def generate_data():
    model = NoackModel(mu=0.1, omega=1.0, A=-0.1, lam=10)
    dt = 0.1
    step = model.get_stepper(dt, method="rk2")

    # generate trajectories for training/testing
    num_train = 1024
    num_test = 64
    n = 200  # length of each trajectory
    print("Generating training and testing trajectories...")
    training_traj = romnet.sample(step, random_ic, num_train, n)
    test_traj = romnet.sample(step, random_ic, num_test, n)
    training_traj.save("noack_train.traj")
    test_traj.save("noack_test.traj")

    # sample gradients for GAP loss
    s = 10  # samples per trajectory
    L = 20  # horizon for gradient sampling
    print("Sampling gradients...")
    adj_step = model.get_adjoint_stepper(dt, method="rk2")
    training_data, _ = romnet.sample_gradient_long_traj(
        training_traj, adj_step, adj_output, model.num_states, s, L
    )
    test_data, _ = romnet.sample_gradient_long_traj(
        test_traj, adj_step, adj_output, model.num_states, s, L
    )
    print("Done")

    training_data.save("noack_train.dat")
    test_data.save("noack_test.dat")


if __name__ == "__main__":
    generate_data()
