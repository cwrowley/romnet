#!/usr/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np
import romnet
from romnet.models import BarkleyPuffModel
from romnet.models.barkley import fft_multiply, freq_to_space, space_to_freq


def test_to_space():
    nmodes = 16
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    uf[0] = np.real(uf[0])
    u = np.zeros(2 * nmodes, dtype=float)
    x = np.arange(2 * nmodes) * 2 * np.pi / (2 * nmodes)
    u[:] = np.real(uf[0])
    for k in range(1, nmodes):
        u += 2 * np.real(uf[k] * (np.cos(k * x) + 1j * np.sin(k * x)))
    # last Fourier mode is real and appears only once in sum
    u += np.real(uf[-1]) * np.cos(nmodes * x)
    u2 = freq_to_space(uf)
    # plt.plot(x, u)
    # plt.plot(x, u2)
    # plt.show()
    assert np.allclose(u, u2)


def test_to_freq():
    nmodes = 16
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    uf[0] = np.real(uf[0])
    uf[-1] = np.real(uf[-1])
    u = freq_to_space(uf)
    uf2 = space_to_freq(u)
    assert np.allclose(uf, uf2)


def test_fft_mult():
    nmodes = 32
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    vf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    u = freq_to_space(uf)
    v = freq_to_space(vf)
    uv = u * v
    uv2 = freq_to_space(fft_multiply(uf, vf))
    # plt.plot(uv)
    # plt.plot(uv2)
    # plt.show()
    assert np.allclose(uv, uv2)


def main():

    nmodes = 256
    L = 500
    Lt = 500
    dt = 2e-3

    u0 = 2
    ubar = 1
    zeta = 0.8
    D = 0.5
    Du = 0.01
    delta = 0.1
    kappa = 2

    time_interval = 1 # every time units we output the data

    r = 0.75
    sigma = 0.5
    epsilon = 0.01
    filename = "initial_condition_0.75_0.01.txt"
    job_id = 1256

    # r = float(sys.argv[1])
    # sigma = float(sys.argv[2])
    # epsilon = float(sys.argv[3])
    # filename = sys.argv[4]
    # job_id = int(sys.argv[5])

    epsilon_1 = epsilon
    epsilon_2 = kappa * epsilon

    bpm = BarkleyPuffModel(nmodes, L, u0, ubar, zeta, D, Du, delta, epsilon_1, epsilon_2, r, sigma)
    # ks.set_stepper(dt, "rk3cn")
    step = bpm.get_stepper(dt, "setd1")

    # initial condition
    u = np.zeros(2*(nmodes + 1), dtype=complex)
    initial_value  = np.loadtxt(filename)
    q_initial  = initial_value[:,1]
    u_initial  = initial_value[:,2]

    u[:nmodes+1] = space_to_freq(q_initial)
    u[nmodes+1:] = space_to_freq(u_initial)

    nt = int(Lt/dt + 1)
    nx = 2 * nmodes
    dx = L / (2 * nmodes)
    sol = np.zeros((nt, 2 * nx))
    T = np.arange(nt) * dt
    x = np.arange(2 * nmodes) * dx

    path = "/scratch/network/ys5910/Barkley_Model/Stochastic/" + "solution_r_" + str(r) + "_sigma_" + str(sigma) + "_epsilon_" + str(epsilon_1) + "_no_" + str(job_id)
    os.makedirs( path )
    os.chdir( path )

    print("Computing solution...")

    for t in range(nt):
        u = step(u)
        sol[t] = np.append(freq_to_space(u[:nmodes+1]),freq_to_space(u[nmodes+1:]))
        if t % int(time_interval/dt) == 0:
            print("  step %4d / %4d" % (t, nt))

            filename = "no_" + str(job_id) + "_time_" + str( int(t*dt) ) + ".txt"
            np.savetxt( filename , np.column_stack( (x, sol[t,:nx], sol[t,nx:]) ), fmt = '%.4f %e %e')
            #np.savetxt( filename , np.column_stack( (x, sol[:nx], sol[nx:]) ), fmt = '%.4f %e %e')

            plt.plot(x, sol[t,:nx], color = 'r', label='q')
            plt.plot(x, sol[t,nx:], color = 'b', label='u')
            #plt.plot(x, sol[:nx], color = 'r', label='q')
            #plt.plot(x, sol[nx:], color = 'b', label='u')
            plt.xlabel("x")
            plt.ylim(0, u0+3*sigma)
            plt.title(f"Variables in Barkley's no.{job_id} DNS, r = {r}, $ \sigma $ = {sigma}, t = {int(t*dt)}, {nmodes} modes")
            plt.legend(loc = "upper right")
            plt.savefig( "Variation_no_" + str(job_id) + "_r_" + str(r) + "_sigma_" + str(sigma) + "_time_" + str( int(t*dt) )  + ".jpg", dpi=144)
            plt.clf()

    fig2, ax2 = plt.subplots()
    fig_time_evolution_q = ax2.contourf(x, T, sol[:,:nx])
    ax2.set_xlim(0, L)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title(f"Time Evolution of r = {r}, $ \sigma $ = {sigma}, $ \epsilon $ = {epsilon}, {nmodes} modes")
    fig2.colorbar(fig_time_evolution_q)
    plt.savefig( "Evolution_no_" + str(job_id) + "_r_" + str(r) + "_sigma_" + str(sigma) + "_epsilon_" + str(epsilon) + ".jpg", dpi=1200)
    plt.show()

if __name__ == "__main__":
    main()
