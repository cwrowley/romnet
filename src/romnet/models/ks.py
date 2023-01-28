"""Kuramoto-Sivashinsky equation"""

import numpy as np
import scipy as sp

from .. import BilinearModel

__all__ = ["KuramotoSivashinsky", "freq_to_space", "space_to_freq"]


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


class KuramotoSivashinsky(BilinearModel):
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
