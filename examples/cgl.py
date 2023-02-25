#!/usr/bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np
import romnet
from romnet.models import CGL
from scipy.integrate import solve_ivp


def compare_timesteppers():
    t_final = 500
    nx = 220
    dt1 = 0.5
    dt2 = 0.5
    dt3 = 0.5
    num_steps1 = int(t_final / dt1) + 1
    num_steps2 = int(t_final / dt2) + 1
    num_steps3 = int(t_final / dt3) + 1
    t1 = dt1 * np.arange(num_steps1)
    t2 = dt2 * np.arange(num_steps2)
    t3 = dt3 * np.arange(num_steps3)

    model = CGL(nx)
    method1 = "rk3cn"
    step1 = model.get_stepper(dt1, method=method1)

    method2 = "rk2cn"
    step2 = model.get_stepper(dt2, method=method2)

    q0 = model.random_ic()

    q1 = np.zeros((num_steps1, model.num_states))
    q1[0] = q0
    tic = time.time()
    for i in range(num_steps1 - 1):
        q1[i + 1] = step1(q1[i])
    toc = time.time()
    print(f"Elapsed time for {method1} = {toc - tic}")

    q2 = np.zeros((num_steps2, model.num_states))
    q2[0] = q0
    tic = time.time()
    for i in range(num_steps2 - 1):
        q2[i + 1] = step2(q2[i])
    toc = time.time()
    print(f"Elapsed time for {method2} = {toc - tic}")

    tic = time.time()
    sol = solve_ivp(
        lambda _, q: model.rhs(q),
        # jac=lambda t, q: model.jac(q),
        t_span=[0, t_final],
        y0=q0,
        t_eval=t3,
        method="BDF",
    )
    toc = time.time()
    print(f"Elapsed time for solve_ivp = {toc - tic}")

    y1 = model.output(q1.T).T
    y2 = model.output(q2.T).T
    y3 = model.output(sol.y_traj).T
    plt.figure()
    plt.plot(t1, y1[:, 0], label=method1)
    plt.plot(t2, y2[:, 0], label=method2)
    plt.plot(t3, y3[:, 0], label="solve_ivp")
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()


def generate_data():

    np.random.seed(seed=0)

    # getting discrete model stepper
    model = CGL()
    dt = 0.1
    step = model.get_stepper(dt, method="rk3cn")

    # generating trajectories
    num_train = 100
    num_test = 10
    t_final = 100
    n = int(t_final / dt) + 1
    t = dt * np.arange(n)
    print("Generating training trajectories...")
    training_traj = romnet.sample(step, model.random_ic, num_train, n)
    test_traj = romnet.sample(step, model.random_ic, num_test, n)
    training_traj.save("cgl_train.traj")
    test_traj.save("cgl_test.traj")

    # sampling gradients
    s = 10   # samples per trajectory
    L = 500  # horizon for gradient sampling
    adj_step = model.get_adjoint_stepper(dt, method="rk3cn")
    print("Sampling gradients...")
    training_data, _ = romnet.sample_gradient_long_traj(
        training_traj, adj_step, model.adjoint_output, model.num_outputs, s, L
    )
    test_data, _ = romnet.sample_gradient_long_traj(
       test_traj, adj_step, model.adjoint_output, model.num_outputs, s, L
    )

    # use CoBRAS-like rom to determine rank of projection
    print("Calculating CoBRAS projectors...")
    X = training_data.X[::100]
    G = training_data.G[::100]
    cobras = romnet.CoBRAS(X, G)
    cobras.save_projectors("cgl.cobras")
    rank = 15
    Phi, Psi = cobras.projectors()
    Phi = Phi[:, :rank]
    Psi = Psi[:, :rank]
    rom = model.project(Phi.T, Psi.T)
    rom_step = rom.get_stepper(dt, method="rk3cn")
    z_ics = test_traj.traj[:, 0, :] @ Psi
    z_ics_iter = iter(z_ics)

    def z_ic():
        return next(z_ics_iter)
    z_rom_traj = romnet.sample(rom_step, z_ic, num_test, n)
    y_traj = test_traj.traj @ model.C.T
    y_rom_traj = z_rom_traj.traj @ Phi.T @ model.C.T
    error = np.square(np.linalg.norm(y_traj - y_rom_traj, axis=2))
    E_avg = np.mean(np. square(np.linalg.norm(y_traj, axis=2)))
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.semilogy(t, (error / E_avg).T)
    ax.set_ylabel("$\\frac{|| y_{rom} - y_traj ||}{E_{avg}}$")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(t, y_traj[0, :, 0], label="FOM")
    ax.plot(t, y_rom_traj[0, :, 0], label="ROM", linestyle='--')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$Re(y_traj)$")
    ax.legend()
    plt.show()

    # ProjectedGradientDataset
    print("Generating the projected gradient dataset...")
    reduced_training_data = cobras.project(training_data.X, training_data.G, rank)
    reduced_testing_data = cobras.project(test_data.X, test_data.G, rank)
    reduced_training_data.save("cgl_train.dat")
    reduced_testing_data.save("cgl_test.dat")

    print("Done")


if __name__ == "__main__":
    # compare_timesteppers()
    generate_data()
