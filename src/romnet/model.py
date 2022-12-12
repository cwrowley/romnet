#!/usr/bin/env python
"""model - Define how a given state evolves in time."""

import abc
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import torch

__all__ = ["Model", "SemiLinearModel", "BilinearModel"]


class Timestepper(abc.ABC):
    """Abstract base class for timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method):
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError(f"Method '{method}' unknown") from exc

    def __init__(self, dt, rhs, adjoint_rhs=None, nsteps=1):
        self.dt = dt
        self.rhs = rhs
        self.adjoint_rhs = adjoint_rhs
        self.nsteps = nsteps

    @abc.abstractmethod
    def step_(self, x, rhs):
        """Advance the state x by one timestep, for the ODE x' = rhs(x)."""

    def step(self, x):
        for _ in range(self.nsteps):
            x = self.step_(x, self.rhs)
        return x

    def adjoint_step(self, x, v):
        def f(v):
            return self.adjoint_rhs(x, v)
        for _ in range(self.nsteps):
            v = self.step_(v, f)
        return v

    @classmethod
    def methods(cls):
        return list(cls.__registry.keys())


class SemiImplicit(abc.ABC):
    """Abstract base class for semi-implicit timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method):
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError(f"Method '{method}' unknown") from exc

    def __init__(self, dt, linear, nonlinear, adjoint_nonlinear, nsteps=1):
        self._dt = dt
        self.linear = linear
        self.nonlinear = nonlinear
        self.adjoint_nonlinear = adjoint_nonlinear
        self.nsteps = nsteps
        self.update()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        self.update()

    @abc.abstractmethod
    def update(self):
        """Update quantities used in the semi-implicit solve.

        This routine is called when the timestepper is created, and whenever
        the timestep is changed
        """

    @abc.abstractmethod
    def step_(self, x, linear, nonlinear, transpose=0):
        """Advance the state forward by one step"""

    def step(self, x):
        """Advance the state x by one step."""
        for _ in range(self.nsteps):
            x = self.step_(x, self.linear, self.nonlinear, transpose=0)
        return x

    def adjoint_step(self, x, v):
        """Advance the adjoint variable v by one step, linearized about x."""
        def f(w):
            return self.adjoint_nonlinear(x, w)
        for _ in range(self.nsteps):
            v = self.step_(v, self.linear.T, f, transpose=1)
        return v

    @classmethod
    def methods(cls):
        return list(cls.__registry.keys())


class Euler(Timestepper):
    """Explicit Euler timestepper."""

    def step_(self, x, rhs):
        return x + self.dt * rhs(x)


class RK2(Timestepper):
    """Second-order Runge-Kutta timestepper."""

    def step_(self, x, rhs):
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1)
        return x + (k1 + k2) / 2.


class RK4(Timestepper):
    """Fourth-order Runge-Kutta timestepper."""

    def step_(self, x, rhs):
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1 / 2.)
        k3 = self.dt * rhs(x + k2 / 2.)
        k4 = self.dt * rhs(x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.


class RK2CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 2nd-order Runge-Kutta.

    See Peyret p148-149
    """

    def update(self):
        nstates = self.linear.shape[0]
        mat = np.eye(nstates) - 0.5 * self.dt * self.linear
        self.LU = lu_factor(mat)

    def step_(self, x, linear, nonlinear, transpose=0):
        rhs_linear = x + 0.5 * self.dt * np.dot(linear, x)
        Nx = nonlinear(x)

        rhs1 = rhs_linear + self.dt * Nx
        x1 = lu_solve(self.LU, rhs1, trans=transpose)

        rhs2 = rhs_linear + 0.5 * self.dt * (Nx + nonlinear(x1))
        x2 = lu_solve(self.LU, rhs2, trans=transpose)
        return x2


class RK3CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 3rd-order Runge-Kutta.

    Peyret, p.146 and 149
    """

    A = [0, -5./9, -153./128]
    B = [1./3, 15./16, 8./15]
    Bprime = [1./6, 5./24, 1./8]

    def update(self):
        nstates = self.linear.shape[0]
        mat1 = np.eye(nstates) - self.Bprime[0] * self.dt * self.linear
        mat2 = np.eye(nstates) - self.Bprime[1] * self.dt * self.linear
        mat3 = np.eye(nstates) - self.Bprime[2] * self.dt * self.linear

        self.LU = []
        self.LU.append(lu_factor(mat1))
        self.LU.append(lu_factor(mat2))
        self.LU.append(lu_factor(mat3))

    def step_(self, x, linear, nonlinear, transpose=0):
        A = self.A
        B = self.B
        Bprime = self.Bprime

        Q1 = self.dt * nonlinear(x)
        rhs1 = x + B[0] * Q1 + Bprime[0] * self.dt * np.dot(linear, x)
        x1 = lu_solve(self.LU[0], rhs1, trans=transpose)

        Q2 = A[1] * Q1 + self.dt * nonlinear(x1)
        rhs2 = x1 + B[1] * Q2 + Bprime[1] * self.dt * np.dot(linear, x1)
        x2 = lu_solve(self.LU[1], rhs2, trans=transpose)

        Q3 = A[2] * Q2 + self.dt * nonlinear(x2)
        rhs3 = x2 + B[2] * Q3 + Bprime[2] * self.dt * np.dot(linear, x2)
        x3 = lu_solve(self.LU[2], rhs3, trans=transpose)
        return x3


class Model(abc.ABC):
    """Abstract base class defining an ODE dx/dt = f(x).

    Subclasses must override two methods:
      rhs(x) - returns the right-hand side f(x)
      adjoint_rhs(x, v) - returns the adjoint of Df(x), applied to the vector v
    """

    @abc.abstractmethod
    def rhs(self, x):
        """Return the right-hand-side of the ODE x' = f(x)."""

    @abc.abstractmethod
    def adjoint_rhs(self, x, v):
        """For the right-hand-side function f(x), return Df(x)^T v."""

    def output(self, x):
        """
        Return output y = g(x).

        Default output is y = x
        """
        return x

    def adjoint_output(self, x, v):
        """
        For output y = g(x), return Dg(x)^T v.

        Default output is y = x
        """
        return v

    def set_stepper(self, dt, method="rk2", **kwargs):
        """Configure a timestepper for the model."""
        cls = Timestepper.lookup(method)
        self.stepper = cls(dt, self.rhs,
                           adjoint_rhs=self.adjoint_rhs, **kwargs)
        self.step = self.stepper.step
        self.adjoint_step = self.stepper.adjoint_step


class SemiLinearModel(Model):
    """Abstract base class for semi-linear models.

    Subclasses describe a model of the form
        x' = A x + N(x)
    """

    @abc.abstractmethod
    def nonlinear(self, x):
        pass

    @abc.abstractmethod
    def adjoint_nonlinear(self, x, v):
        pass

    def rhs(self, x):
        return np.dot(self.linear, x) + self.nonlinear(x)

    def adjoint_rhs(self, x, v):
        return np.dot(self.linear.T, v) + self.adjoint_nonlinear(x, v)

    def set_stepper(self, dt, method="rk2cn", **kwargs):
        """Configure a timestepper for the semilinear model."""
        if method in SemiImplicit.methods():
            cls = SemiImplicit.lookup(method)
            self.stepper = cls(dt, self.linear, self.nonlinear,
                               self.adjoint_nonlinear, **kwargs)
            self.step = self.stepper.step
            self.adjoint_step = self.stepper.adjoint_step
        else:
            super().set_stepper(dt, method=method, **kwargs)


class BilinearModel(SemiLinearModel):
    """Model where the right-hand side is a bilinear function of the state

    Models have the form
        x' = c + L x + B(x, x)
    where B is bilinear
    """

    def __init__(self, c, L, B):
        self.affine = c
        self.linear = L
        self._bilinear = B

    def bilinear(self, a, b):
        """Evaluate the bilinear term B(a, b)"""
        return self._bilinear.dot(a).dot(b)

    def nonlinear(self, x):
        return self.affine + self.bilinear(x, x)

    def adjoint_nonlinear(self, x, v):
        w = np.einsum("kji, j, k", self._bilinear, x, v)
        w += np.einsum("jik, j, k", self._bilinear, v, x)
        return w

    def project(self, V, W=None):
        """
        Return a reduced-order model by projecting onto linear subspaces

        Rows of V determine the subspace to project onto
        Rows of W determine the direction of projection

        That is, the projection is given by
          V' (WV')^{-1} W

        The number of states in the reduced-order model is the number of rows
        in V (or W).

        If W is not specified, it is assumed W = V
        """
        numstates = len(V)
        if W is None:
            W = V
        assert len(W) == numstates

        # Let W1 = (W V')^{-1} W
        G = W @ V.T
        W1 = np.linalg.solve(G, W)
        # Now projection is given by P = V' W1, and W1 V' = Identity

        c = W1 @ self.affine
        L = W1 @ self.linear @ V.T
        B = np.zeros((numstates, numstates, numstates))
        for i in range(numstates):
            for j in range(numstates):
                for k in range(numstates):
                    B[i, j, k] = np.dot(W1[i], self.bilinear(V[j], V[k]))
        return BilinearModel(c, L, B)


class LinearLiftedROM(Model):
    """
    Return a reduced-order model by projecting onto linear subspaces

    Rows of V determine the subspace to project onto
    Rows of W determine the direction of projection

    That is, the projection is given by
        V' (WV')^{-1} W

    The number of states in the reduced-order model is the number of rows
    in V (or W).

    If W is not specified, it is assumed W = V
    """

    def __init__(self, model, V, W=None):
        self.model = model
        self.V = V
        numstates = len(V)
        if W is None:
            W = V
        assert len(W) == numstates

        # Let W1 = (W V')^{-1} W
        G = W @ V.T
        self.W1 = np.linalg.solve(G, W)
        # Now projection is given by P = V' W1, and W1 V' = Identity

    def rhs(self, z):
        return self.W1 @ self.model.rhs(self.V.T @ z)

    def adjoint_rhs(self, x, v):
        return -1


class NetworkLiftedROM(Model):
    """
    Return a reduced-order model by projecting onto the tangent space
    of a nonlinear manifold defined by the range of a romnet neural network.

    Torch gradient information is not preserved.

    The rom neural network is a differentiable idempotent operator
    P(x) = psid(psie(x)) = psid(z).

    The reduced-order model in state space is given by
    xdot = DP(x)f(x).

    The reduced-order model in the latent space is given by
    zdot = Dpsie(psid(z))f(psid(z)).
    """

    def __init__(self, model, autoencoder):
        self.model = model
        self.autoencoder = autoencoder

    def rhs(self, z):
        with torch.no_grad():
            x = self.autoencoder.dec(z)
            _, v = self.autoencoder.d_enc(x, self.model.rhs(x))
            return v.numpy()

    def adjoint_rhs(self, x, v):
        return -1
