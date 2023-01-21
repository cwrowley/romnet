#!/usr/bin/env python

import numpy as np
import scipy as sp

import romnet


def fft_multiply(yf, zf):
    """Multiply two Fourier series using fast Fourier transform"""
    nmodes = len(yf) - 1
    y = sp.fft.irfft(yf)
    z = sp.fft.irfft(zf)
    yz = y * z
    return 2 * nmodes * sp.fft.rfft(yz)


def freq_to_space(yf):
    nmodes = len(yf) - 1
    return 2 * nmodes * sp.fft.irfft(yf)


def space_to_freq(y):
    nmodes = len(y) // 2
    return sp.fft.rfft(y) / (2 * nmodes)


class KuramotoSivashinsky(romnet.BilinearModel):
    r"""Kuramoto-Sivashinsky equation

    u_t + u u_x + u_xx + u_xxxx = 0

    with periodic boundary conditions, for 0 <= x <= L
    The equation is solved using a spectral collocation or spectral Galerkin
    method

    .. math::
        u(x,t) = \sum_{k=-n}^n u_k(t) \exp(2\pi i k x / L)

    Since u is real, this implies :math:`u_{-k} = \overline{u_k}`

    The state is represented as a vector of Fourier coefficients u_k,
    for k = 0, ..., nmodes.
    """

    def __init__(self, nmodes: int, L: float):
        """summary here

        Args:
            nmodes: Number of (complex) Fourier modes to use
            L: Length of domain
        """
        self.nmodes = nmodes
        self.L = L
        k = np.arange(nmodes + 1)
        ksq = (2 * np.pi / L) ** 2 * k**2
        # Linear part = -u_xx - u_xxxx
        self._linear_factor = ksq * (1 - ksq)
        self._deriv = (2j * np.pi / L) * k

    def get_solver(self, alpha: float):
        def solver(u):
            return u / (1 - alpha * self._linear_factor)

        return solver

    def get_adjoint_solver(self, alpha: float):
        # linear part is self-adjoint
        return self.get_solver(alpha)

    def linear(self, u):
        return u * self._linear_factor

    def adjoint(self, u):
        # linear part is self-adjoint
        return self.linear(u)

    def bilinear(self, a, b):
        # a * b_x
        return fft_multiply(a, self._deriv * b)

    def nonlinear(self, u):
        return self.bilinear(u, u)

    def adjoint_nonlinear(self, u, w):
        return self.bilinear(w, u) - self.bilinear(u, w)


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
    import matplotlib.pyplot as plt

    nmodes = 512
    L = 400
    ks = KuramotoSivashinsky(nmodes, L)
    dt = 0.1
    ks.set_stepper(dt, "rk3cn")

    # initial condition
    u = np.zeros(nmodes + 1, dtype=complex)
    u[45] = 0.9
    u[46] = 1j

    nt = 5000
    nx = 2 * nmodes
    sol = np.zeros((nt, nx))
    print("Computing solution...")
    for t in range(nt):
        u = ks.step(u)
        sol[t] = freq_to_space(u)
        if t % 1000 == 0:
            print("  step %4d / %4d" % (t, nt))
    T = np.arange(nt) * dt
    dx = L / (2 * nmodes)
    x = np.arange(2 * nmodes) * dx

    fig, ax = plt.subplots()
    ax.contourf(x, T, sol)
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({nmodes} modes)")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, nmodes + 1), np.abs(u[1:]))
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Fourier coefficient magnitude")
    ax.set_title("Energy spectrum")
    plt.show()


if __name__ == "__main__":
    main()
