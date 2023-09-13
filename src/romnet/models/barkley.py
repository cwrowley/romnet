"""Barkley model for turbulent puffs

D. Barkley, Theoretical perspective on the route to turbulence in a pipe,
Journal of Fluid Mechanics 803 P1, 2016
"""

from typing import Callable

import numpy as np

from .. import SemiLinearModel
from ..typing import Vector
from .ks import fft_multiply, freq_to_space, space_to_freq

__all__ = ["BarkleyPuffModel", "freq_to_space", "space_to_freq"]


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
        self.N = nmodes + 1     # number of Fourier modes in each dependent variable

        # Linear dynamics
        self._linear_tl = -(delta + u0) * np.ones(self.N) - D * ksq + zeta * self._deriv
        self._linear_bl = epsilon_2 * ubar * np.ones(self.N)
        self._linear_br = -epsilon_1 * np.ones(self.N) - Du * ksq

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        def solver(u: Vector) -> Vector:
            return u / (1 - alpha * self._linear_factor)

        return solver

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        # linear part is self-adjoint
        return self.get_solver(alpha)

    def linear(self, state: Vector) -> Vector:
        q = state[:self.N]
        u = state[self.N:]
        linear_q = self._linear_tl * q
        linear_u = self._linear_bl * q + self._linear_br * u
        return np.append(linear_q, linear_u)

    def adjoint(self, u: Vector) -> Vector:
        # linear part is self-adjoint
        return self.linear(u)

    def bilinear(self, a: Vector, b: Vector) -> Vector:
        # a * b_x
        return fft_multiply(a, self._deriv * b)

    def nonlinear(self, state: Vector) -> Vector:
        q = state[:self.N]
        u = state[self.N:]
        q_sq = fft_multiply(q, q)
        q_nonlinear = (fft_multiply(u, q - self._deriv * q)
                       - (self.r + self.delta) * (fft_multiply(q_sq, q) - 2 * q_sq))
        u_nonlinear = (space_to_freq(self.epsilon_1 * self.u0 * np.ones(2 * self.nmodes))
                       - self.epsilon_2 * fft_multiply(u, q)
                       - fft_multiply(u, self._deriv * u))
        return np.append(q_nonlinear, u_nonlinear)

    def adjoint_nonlinear(self, u: Vector, w: Vector) -> Vector:
        return self.bilinear(w, u) - self.bilinear(u, w)

    def noise(self, state: Vector) -> Vector:
        q = state[:self.N]
        noise_mu = 0.0
        noise_sig = 1.0
        noise_npts = 2 * self.nmodes
        white_noise = space_to_freq(np.random.normal(noise_mu, noise_sig, noise_npts))
        multiple_noise = fft_multiply(white_noise, q)
        return self.sigma * np.append(multiple_noise, np.zeros(self.N))

    def linear_exp(self, state: Vector, alpha: float) -> Vector:
        q = state[:self.N]
        u = state[self.N:]
        expL_q = np.exp(self._linear_tl * alpha) * q
        expL_u = (np.exp(self._linear_br * alpha) * u
                  + q * self._linear_bl
                  * (np.exp(alpha * self._linear_tl) - np.exp(alpha * self._linear_br))
                  / (self._linear_tl - self._linear_br))
        return np.append(expL_q, expL_u)

    def matrix_inv(self, state: Vector) -> Vector:
        q = state[:self.N]
        u = state[self.N:]
        sol_q = q / self._linear_tl
        sol_u = (u - q * self._linear_bl / self._linear_tl) / self._linear_br
        return np.append(sol_q, sol_u)
