
import numpy as np

import romnet
from romnet.models import Pitchfork
from romnet.typing import Vector


def random_ic():
    xmax = 6
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    return np.array((x, y))


def adj_output(x: Vector, eta: Vector) -> Vector:
    return eta


def generate_data():
    model = Pitchfork()
    t_final = 800
    dt = 0.1
    step = model.get_stepper(dt, method="rk4")

    # generate trajectories for training/testing
    num_train = 1024
    num_test = 64
    n = int(t_final / dt) + 1  # length of each trajectory
    print("Generating training and testing trajectories...")
    training_traj = romnet.sample(step, random_ic, num_train, n)
    test_traj = romnet.sample(step, random_ic, num_test, n)
    training_traj.save("pitchfork_train.traj")
    test_traj.save("pitchfork_test.traj")

    # sample gradients for GAP loss
    s = 10  # samples per trajectory
    L = 20  # horizon for gradient sampling
    print("Sampling gradients...")
    adj_step = model.get_adjoint_stepper(dt, method="rk4")
    training_data, _ = romnet.sample_gradient_long_traj(
        training_traj, adj_step, adj_output, model.num_states, s, L
    )
    test_data, _ = romnet.sample_gradient_long_traj(
        test_traj, adj_step, adj_output, model.num_states, s, L
    )
    print("Done")

    training_data.save("pitchfork_train.dat")
    test_data.save("pitchfork_test.dat")


if __name__ == "__main__":
    generate_data()
