"""Kuramoto-Sivashinsky equation"""

from typing import Callable

import numpy as np
import scipy as sp

from .. import SemiLinearModel
from ..typing import Vector

__all__ = ["BarkleyPuffModel", "freq_to_space", "space_to_freq"]


def fft_multiply(yf: Vector, zf: Vector) -> Vector:
    """Multiply two Fourier series using fast Fourier transform"""
    nmodes = len(yf) - 1
    y = sp.fft.irfft(yf)
    z = sp.fft.irfft(zf)
    yz = y * z
    return 2 * nmodes * sp.fft.rfft(yz)


def freq_to_space(yf: Vector) -> Vector:
    nmodes = len(yf) - 1
    return 2 * nmodes * sp.fft.irfft(yf)


def space_to_freq(y: Vector) -> Vector:
    nmodes = len(y) // 2
    return sp.fft.rfft(y) / (2 * nmodes)


class BarkleyPuffModel(SemiLinearModel):

    def __init__(self, nmodes: int, L: float, u0: float, ubar: float, zeta: float, D:float, Du:float, delta: float, epsilon_1: float, epsilon_2: float, r: float, sigma: float):
        """summary here

        Args:
            nmodes: Number of (complex) Fourier modes to use
            L: Length of domain
        """
        self.nmodes = nmodes
        self.L = L
        self.ubar = ubar
        self.u0 = u0
        self.zeta = zeta
        self.D = D
        self.Du = Du
        self.delta = delta
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.r = r
        self.sigma = sigma

        k = np.arange(nmodes + 1)
        ksq = (2 * np.pi / L) ** 2 * k**2
        self._deriv = (2j * np.pi / L) * k

        # Linear dynamics
        self._linear_factor_tl = -(delta+u0)*np.ones(self.nmodes+1) - D*ksq + zeta*self._deriv
        self._linear_factor_bl = epsilon_2*ubar*np.ones(self.nmodes+1)
        self._linear_factor_br = -epsilon_1*np.ones(self.nmodes+1) - Du*ksq


    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        def solver(u: Vector) -> Vector:
            return u / (1 - alpha * self._linear_factor)

        return solver

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        # linear part is self-adjoint
        return self.get_solver(alpha)

    def linear(self, u: Vector) -> Vector:
        linear_component_q = self._linear_factor_tl*u[:self.nmodes+1]
        linear_component_u = self._linear_factor_bl*u[:self.nmodes+1] + self._linear_factor_br*u[self.nmodes+1:]
        return np.append(linear_component_q, linear_component_u)

    def adjoint(self, u: Vector) -> Vector:
        # linear part is self-adjoint
        return self.linear(u)

    def bilinear(self, a: Vector, b: Vector) -> Vector:
        # a * b_x
        return fft_multiply(a, self._deriv * b)

    def nonlinear(self, u: Vector) -> Vector:
        q_nonlinear = fft_multiply(u[self.nmodes+1:], u[:self.nmodes+1]-self._deriv*u[:self.nmodes+1]) - (self.r+self.delta)*(fft_multiply(fft_multiply(u[:self.nmodes+1],u[:self.nmodes+1]), u[:self.nmodes+1]) - 2*fft_multiply(u[:self.nmodes+1],u[:self.nmodes+1]))
        u_nonlinear = space_to_freq(self.epsilon_1*self.u0*np.ones(2*self.nmodes)) - self.epsilon_2*fft_multiply(u[self.nmodes+1:], u[:self.nmodes+1]) - fft_multiply(u[self.nmodes+1:], self._deriv*u[self.nmodes+1:])
        return np.append(q_nonlinear, u_nonlinear)

    def adjoint_nonlinear(self, u: Vector, w: Vector) -> Vector:
        return self.bilinear(w, u) - self.bilinear(u, w)

    def noise(self, u: Vector) -> Vector:
        noise_mu = 0
        noise_sig = 1
        noise_size = 2*self.nmodes
        white_noise = space_to_freq(np.random.normal(noise_mu, noise_sig, noise_size))
        multiple_noise = fft_multiply(white_noise, u[:self.nmodes+1])
        return self.sigma*np.append(multiple_noise, np.zeros(self.nmodes+1))

    def linear_exp(self, u: Vector, alpha: float) -> Vector:
        linear_component_q = np.exp(self._linear_factor_tl*alpha)*u[:self.nmodes+1]
        linear_component_u = np.exp(self._linear_factor_br*alpha)*u[self.nmodes+1:] + u[:self.nmodes+1]*self._linear_factor_bl*(np.exp(alpha*self._linear_factor_tl)-np.exp(alpha*self._linear_factor_br))/(self._linear_factor_tl - self._linear_factor_br)
        return np.append(linear_component_q, linear_component_u)

    def matrix_inv(self, u: Vector) -> Vector:
        invAu_q_component = u[:self.nmodes+1]/self._linear_factor_tl
        invAu_u_component = (u[self.nmodes+1:] - (u[:self.nmodes+1]*self._linear_factor_bl/self._linear_factor_tl))/self._linear_factor_br
        return np.append(invAu_q_component, invAu_u_component)
