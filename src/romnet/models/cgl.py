"""Complex Ginzburg-Landau model"""

from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_hermite, gammaln, roots_hermite

from .. import LUSolver, SemiLinearModel
from ..typing import Vector

__all__ = ["CGL"]


def herdif(N: int, order: int) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:  # noqa: C901
    # define collocation points
    x_coll, _ = roots_hermite(N)

    # define constants c using equation (5 in W&R)
    c = np.exp(-np.square(x_coll) / 2.0)
    for j in range(N):
        for k in range(j):
            c[j] = c[j] * (x_coll[j] - x_coll[k])

        for k in range(j + 1, N):
            c[j] = c[j] * (x_coll[j] - x_coll[k])

    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = c[i] / c[j]

    # compute beta's using equation (18 in W&R)
    B = np.zeros((order, N))
    bn1 = np.zeros(N)
    b0 = np.ones(N)
    for k in range(order):
        B[k, :] = -x_coll * b0 - k * bn1
        bn1 = np.copy(b0)
        b0 = np.copy(B[k, :])

    # define matrix Z
    Z = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            Z[i, j] = 1.0 / (x_coll[i] - x_coll[j])
            Z[j, i] = 1.0 / (x_coll[j] - x_coll[i])

    # define matrix X
    X = np.zeros((N - 1, N))
    for i in range(N - 1):
        for j in range(i + 1):
            X[i, j] = 1.0 / (x_coll[j] - x_coll[i + 1])

        for j in range(i + 1, N):
            X[i, j] = 1.0 / (x_coll[j] - x_coll[i])

    # main recursion described in Table II of W&R
    Y = np.ones((N - 1, N))
    D = np.eye(N)
    D_mats = list()
    for k in range(order):
        # recursion for diagonals
        Y = np.cumsum(np.vstack([np.reshape(B[k, :], (1, -1)),
                                 (k + 1) * Y[:N - 1, :] * X]), axis=0)

        # recursion for off-diagonals
        D_rep = np.reshape(np.diag(D), (-1, 1)) * np.ones((N, N))
        D = (k + 1) * Z * (C * D_rep - D)

        # correct the diagonal of D
        for i in range(N):
            D[i, i] = Y[N - 1, i]

        D_mats.append(D)

    return x_coll, D_mats


class CGL(SemiLinearModel):
    """
    Complex Ginzburg-Landau model
    """

    def __init__(self, nx: int = 220, U: float = 2.0, cu: float = 0.2, cd: float = -1.0,
                 mu0: float = 0.41, mu2: float = -0.01, a: float = 0.1):
        self.nx = nx
        self.U = U
        self.cu = cu
        self.cd = cd
        self.mu0 = mu0
        self.mu2 = mu2
        self.a = a
        self._linear, self.weights, self.xi, self.x = \
            self._construct_matrices()
        self.C = self._construct_output()

    @property
    def num_states(self) -> int:
        return 2 * self.nx

    @property
    def num_outputs(self) -> int:
        return self.C.shape[0]

    @property
    def gamma(self) -> complex:
        return 1.0 + 1j * self.cd

    @property
    def chi(self) -> float:
        return np.power(np.absolute(self.mu2 / (2 * self.gamma)), 0.25)

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.mu0 - self.cu**2 + 0.5 * self.mu2 * x**2

    def _construct_matrices(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # collocation points and differentiation operators
        xi, Dxi_mats = herdif(self.nx, 2)
        # quadrature weights
        _, w = roots_hermite(self.nx)
        chi = self.chi
        chi_sq = np.square(chi)
        w_quad = w * np.exp(xi**2) / chi

        # evaluate mu at collocation points
        x = xi / chi
        mu_mat = np.diag(self.mu(x))
        # Linear operators for CGL in state space with weighted inner product
        U = self.U
        cu = self.cu
        cd = self.cd
        Lrr = -U * chi * Dxi_mats[0] + chi_sq * Dxi_mats[1] + mu_mat
        Lri = 2 * cu * chi * Dxi_mats[0] - cd * chi_sq * Dxi_mats[1]
        Lir = -2 * cu * chi * Dxi_mats[0] + cd * chi_sq * Dxi_mats[1]
        Lii = -U * chi * Dxi_mats[0] + chi_sq * Dxi_mats[1] + mu_mat
        # Linear operators for CGL in Euclidean state space
        w_col = np.reshape(np.sqrt(w_quad), (-1, 1))
        Arr = w_col * Lrr / w_col.T
        Ari = w_col * Lri / w_col.T
        Air = w_col * Lir / w_col.T
        Aii = w_col * Lii / w_col.T
        A = np.vstack([np.hstack([Arr, Ari]), np.hstack([Air, Aii])])
        return A, w_quad, xi, x

    def linear(self, x: Vector) -> Vector:
        return self._linear @ x

    def adjoint(self, x: Vector) -> Vector:
        return self._linear.T @ x

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self.num_states) - alpha * self._linear
        return LUSolver(mat)

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self.num_states) - alpha * self._linear.T
        return LUSolver(mat)

    def _construct_output(self) -> NDArray[np.float64]:
        # observation kernel defined by eqn. 4.1 in M. Ilak et al.
        # branch_2 is downstream bound of unstable region
        branch_2 = np.sqrt(-2 * (self.mu0 - np.square(self.cu)) / self.mu2)
        sig = 1.6  # width of the kernel

        ker_obs = np.exp(-np.square((self.xi / self.chi - branch_2) / sig))

        # define observations on the Euclidean state space
        C = np.zeros((2, self.num_states))
        C[0, :self.nx] = ker_obs * np.sqrt(self.weights)
        C[1, self.nx:] = ker_obs * np.sqrt(self.weights)
        return C

    def output(self, q: Vector) -> Vector:
        return np.dot(self.C, q)

    def adjoint_output(self, _: Vector, v: Vector) -> Vector:
        return np.dot(self.C.T, v)

    def nonlinear(self, q: Vector) -> Vector:
        # real and imaginary parts in original state space
        nx = self.nx
        qr = q[:nx]
        qi = q[nx:]

        # evaluate nonlinearity
        f = np.zeros(self.num_states)
        q_sq = (np.square(qr) + np.square(qi)) / self.weights
        f[:nx] = q_sq * qr
        f[nx:] = q_sq * qi
        return -self.a * f

    def adjoint_nonlinear(self, q: Vector, v: Vector) -> Vector:
        """Adjoint of nonlinear term linearized about q, in direction v."""
        # real and imaginary parts of v in Euclidean state space
        nx = self.nx
        vr = v[:nx]
        vi = v[nx:]

        # real and imaginary parts of state in original state space
        qr = q[:nx] / np.sqrt(self.weights)
        qi = q[nx:] / np.sqrt(self.weights)

        # transposed derivative of the nonlinearity
        adjoint = np.zeros(self.num_states)
        adjoint[:nx] = (3 * qr**2 + qi**2) * vr + 2 * qi * qr * vi
        adjoint[nx:] = 2 * qr * qi * vr + (3 * qi**2 + qr**2) * vi
        return -self.a * adjoint

    def random_ic(self, num_modes: int = 10, amplitude: float = 0.01) -> Vector:
        # Use random coefficients on Gaussian Hermite functions
        Psi_vals = np.array([psi_fun(n, self.xi) for n in range(num_modes)]).T
        B_IC = np.zeros((self.num_states, 2 * num_modes))
        B_IC[:self.nx, :num_modes] = Psi_vals
        B_IC[self.nx:, num_modes:] = Psi_vals

        # random initial condition
        q0 = np.dot(B_IC, (amplitude * np.random.randn(2 * num_modes)
                           / np.sqrt(2 * num_modes)))
        return q0


def psi_fun(n: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
    if n < 201:
        Hn = eval_hermite(n, x)
        z = np.square(x) + n * np.log(2) + gammaln(n + 1)
        psi = Hn * np.exp(-z / 2.0)
    else:
        psi0 = np.sqrt(2 / n) * x * psi_fun(n - 1, x)
        psi1 = np.sqrt((n - 1) / n) * psi_fun(n - 2, x)
        psi = psi0 - psi1
    return psi
