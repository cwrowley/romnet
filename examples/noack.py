#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
import romnet
import torch
from romnet.models import NoackModel


def random_ic():
    xmax = 6
    zmin = -6
    zmax = 6
    x = xmax * (2 * np.random.rand() - 1)
    y = xmax * (2 * np.random.rand() - 1)
    z = zmin + (zmax - zmin) * np.random.rand()
    return np.array((x, y, z))


def identity(x):
    return x

def adj_output(x, eta):
    return eta

def generate_data():
    model = NoackModel(mu=0.1, omega=1., A=-0.1, lam=10)
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


def rom(train_num=""):
    # loading data
    print("Loading test trajectories...")
    test_traj = romnet.load("noack_test.traj")

    # romnet
    model = NoackModel(mu=0.1, omega=1., A=-0.1, lam=10)
    autoencoder = romnet.load_romnet("noack" + train_num + ".romnet")
    rom = romnet.NetworkROM(model, autoencoder)

    # generating rom trajectories
    dt = 0.1
    rom.set_stepper(dt, method="rk4")
    with torch.no_grad():
        print("Generating rom trajectories")
        z_ics = autoencoder.enc(test_traj.traj[:, 0, :])
        z_ics_itr = iter(z_ics)

        def z_ic():
            return next(z_ics_itr).numpy()
        z_rom_traj = romnet.sample(rom.step, z_ic, test_traj.num_traj,
                                   test_traj.n)
        print("Done")
        z_rom_traj.save("noack" + train_num + "_rom.traj")


def test_rom(train_num="", savefig=False):
    # loading data
    print("Loading test trajectories...")
    test_traj = romnet.load("noack_test.traj")
    z_rom_traj = romnet.load("noack" + train_num + "_rom.traj")

    # romnet
    model = NoackModel(mu=0.1, omega=1., A=-0.1, lam=10)
    autoencoder = romnet.load_romnet("noack" + train_num + ".romnet")

    with torch.no_grad():

        # normalized l2 error
        x_rom_traj = autoencoder.dec(z_rom_traj.traj).numpy()
        error = np.linalg.norm(test_traj.traj - x_rom_traj, axis=2)
        norm = np.linalg.norm(test_traj.traj, axis=2)
        error_norm = error / norm
        l2error = np.mean(error_norm)
        print("Normalized l2 error: ", l2error)

        # plot slow manifold
        fig = plt.figure()
        xmax = 1.5
        div = 20

        def slow_manifold_input(xmax, div):
            X1 = np.linspace(-xmax, xmax, div)
            X2 = np.linspace(-xmax, xmax, div)
            X = np.zeros((div*div, model.num_states))
            k = 0
            for i in range(div):
                for j in range(div):
                    X[k, 0] = X1[i]
                    X[k, 1] = X2[j]
                    r = np.sqrt(X1[i]**2 + X2[j]**2)
                    X[k, 2] = model.slow_manifold(r)
                    k = k + 1
            return X
        slow_m = slow_manifold_input(xmax, div)
        range_P = autoencoder.forward(slow_m).numpy()

        def surface(X, div):
            X1_graph = np.zeros((div, div))
            X2_graph = np.zeros((div, div))
            X3_graph = np.zeros((div, div))
            k = 0
            for j in range(div):
                for i in range(div):
                    X1_graph[i, j] = X[k, 0]
                    X2_graph[i, j] = X[k, 1]
                    X3_graph[i, j] = X[k, 2]
                    k = k + 1
            return X1_graph, X2_graph, X3_graph
        slow_m_x1, slow_m_x2, slow_m_x3 = surface(slow_m, div)
        range_P_x1, range_P_x2, range_P_x3 = surface(range_P, div)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(slow_m_x1, slow_m_x2, slow_m_x3, color='m', alpha=0.2)
        ax.plot_surface(range_P_x1, range_P_x2, range_P_x3, color='green',
                        alpha=0.3)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        ax.set_xlim([-xmax-1, xmax+1])
        ax.set_ylim([-xmax-1, xmax+1])
        ax.set_zlim([-xmax+1, xmax+3])

        # plot sample test trajectory in full space (Rn)
        cond1 = test_traj.traj[:, 0, 0] <= xmax
        cond2 = test_traj.traj[:, 0, 1] <= xmax
        cond3 = -xmax <= test_traj.traj[:, 0, 0]
        cond4 = -xmax <= test_traj.traj[:, 0, 1]
        traj_num = np.where(cond1 & cond2 & cond3 & cond4)[0][0]
        ax.plot3D(test_traj.traj[traj_num][:, 0],
                  test_traj.traj[traj_num][:, 1],
                  test_traj.traj[traj_num][:, 2], 'blue')
        ax.plot3D(x_rom_traj[traj_num][:, 0],
                  x_rom_traj[traj_num][:, 1],
                  x_rom_traj[traj_num][:, 2], 'red')
        if savefig:
            fig.savefig("noack" + train_num + "_trajRn.pdf", format='pdf')

        # plot sample test trajectory in latent space (Rr)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        z_test = autoencoder.enc(test_traj.traj[traj_num]).numpy()
        z_rom = z_rom_traj.traj[traj_num]
        ax.plot(z_test[:, 0], z_test[:, 1], 'blue')
        ax.plot(z_rom[:, 0], z_rom[:, 1], 'red')
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        if savefig:
            fig.savefig("noack" + train_num + "_trajRr.pdf", format='pdf')

        # plot l2-normalized error time trace
        dt = 0.1
        n = 200
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim([1e-4, 100])
        ax.set_yscale('log')
        t = np.arange(0, n, 1) * dt
        ax.plot(t, error_norm.T, color='blue', linewidth=0.5, alpha=0.5)
        if np.isnan(l2error):
            plt.text(t[-1], 100 * 0.80, r"       $\infty$", fontsize=9)
            plt.plot(t, 90 * np.ones(n), color='blue',
                     linestyle='--', linewidth=2)
        else:
            plt.text(t[-1], l2error * 0.90, "       " + str(100 * l2error) +
                     "%", fontsize=9)
            plt.plot(t, l2error * np.ones(n), color='blue', linestyle='--',
                     linewidth=2)
        if savefig:
            fig.savefig("noack" + train_num + "_error.pdf", format='pdf')

        if not savefig:
            plt.show()


if __name__ == "__main__":
    """
    noack.py --- generate data
    noack.py rom --- generate rom trajectories
    noack.py rom i --- generate rom trajectories for autoencoder i
    noack.py test --- test autoencoder
    noack.py test i --- test autoencoder i
    noack.py test i savefig --- test autoencoder i and save figures
    """
    if len(sys.argv) < 2:
        generate_data()
    elif sys.argv[1] == "rom":
        if len(sys.argv) == 3:
            rom("_" + sys.argv[2])
        else:
            rom()
    elif sys.argv[1] == "test":
        if len(sys.argv) == 3:
            test_rom("_" + sys.argv[2])
        elif (len(sys.argv) == 4) and (sys.argv[3] == "savefig"):
            test_rom("_" + sys.argv[2], savefig=True)
        else:
            test_rom()
