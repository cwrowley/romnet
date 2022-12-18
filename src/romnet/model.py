#!/usr/bin/env python
"""model - Define how a given state evolves in time."""

import abc
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
import torch

__all__ = ["Timestepper", "SemiImplicit",
           "Model", "SemiLinearModel", "BilinearModel"]


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

    def __init__(self, dt):
        self.dt = dt

    @abc.abstractmethod
    def step(self, x, rhs):
        """Advance the state x by one timestep, for the ODE x' = rhs(x)."""

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

    def __init__(self, dt, linear, solver_factory):
        self._dt = dt
        self.linear = linear
        self.get_solver = solver_factory
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
    def step(self, x, nonlinear):
        """Advance the state forward by one step"""

    @classmethod
    def methods(cls):
        return list(cls.__registry.keys())


class Euler(Timestepper):
    """Explicit Euler timestepper."""

    def step(self, x, rhs):
        return x + self.dt * rhs(x)


class RK2(Timestepper):
    """Second-order Runge-Kutta timestepper."""

    def step(self, x, rhs):
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1)
        return x + (k1 + k2) / 2.


class RK4(Timestepper):
    """Fourth-order Runge-Kutta timestepper."""

    def step(self, x, rhs):
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
        self.solve = self.get_solver(0.5 * self.dt)

    def step(self, x, nonlinear):
        rhs_linear = x + 0.5 * self.dt * self.linear(x)
        Nx = nonlinear(x)

        rhs1 = rhs_linear + self.dt * Nx
        x1 = self.solve(rhs1)

        rhs2 = rhs_linear + 0.5 * self.dt * (Nx + nonlinear(x1))
        x2 = self.solve(rhs2)
        return x2


class RK3CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 3rd-order Runge-Kutta.

    Peyret, p.146 and 149
    """

    A = [0, -5./9, -153./128]
    B = [1./3, 15./16, 8./15]
    Bprime = [1./6, 5./24, 1./8]

    def update(self):
        self.solvers = [self.get_solver(b * self.dt) for b in self.Bprime]

    def step(self, x, nonlinear):
        A = self.A
        B = self.B
        Bprime = self.Bprime

        Q1 = self.dt * nonlinear(x)
        rhs1 = x + B[0] * Q1 + Bprime[0] * self.dt * self.linear(x)
        x1 = self.solvers[0](rhs1)

        Q2 = A[1] * Q1 + self.dt * nonlinear(x1)
        rhs2 = x1 + B[1] * Q2 + Bprime[1] * self.dt * self.linear(x1)
        x2 = self.solvers[1](rhs2)

        Q3 = A[2] * Q2 + self.dt * nonlinear(x2)
        rhs3 = x2 + B[2] * Q3 + Bprime[2] * self.dt * self.linear(x2)
        x3 = self.solvers[2](rhs3)
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

    def adjoint_rhs(self, x, v):
        """For the right-hand-side function f(x), return Df(x)^T v."""
        raise NotImplementedError("Adjoint not implemented for class %s" %
                                  self.__class__.__name__)

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

    def step(self, x):
        try:
            return self.stepper.step(x, self.rhs)
        except AttributeError:
            raise AttributeError("Timestepper not defined")

    def adjoint_step(self, x, v):
        def f(w):
            return self.adjoint_rhs(x, w)
        try:
            return self.stepper.step(v, f)
        except AttributeError:
            raise AttributeError("Adjoint timestepper not defined")

    def set_stepper(self, dt, method="rk2", **kwargs):
        """Configure a timestepper for the model."""
        cls = Timestepper.lookup(method)
        self.stepper = cls(dt)


class SemiLinearModel(Model):
    """Abstract base class for semi-linear models.

    Subclasses describe a model of the form
        x' = A x + N(x)
    """

    def linear(self, x):
        return self._linear.dot(x)

    @abc.abstractmethod
    def nonlinear(self, x):
        pass

    def rhs(self, x):
        return self.linear(x) + self.nonlinear(x)

    def adjoint(self, x):
        return self._linear.T.dot(x)

    def adjoint_nonlinear(self, x, v):
        """Return the adjoint of DN(x) applied to the vector v"""
        raise NotImplementedError("Adjoint not implemented for class %s" %
                                  self.__class__.__name__)
        raise AttributeError("Adjoint not implemented")

    def adjoint_rhs(self, x, v):
        return self.adjoint(v) + self.adjoint_nonlinear(x, v)

    def get_solver(self, alpha):
        # default implementation, assumes matrix self._linear has been defined
        nstates = self._linear.shape[0]
        mat = np.eye(nstates) - alpha * self._linear
        return LUSolver(mat)

    def get_adjoint_solver(self, alpha):
        # default implementation, assumes matrix self._linear has been defined
        nstates = self._linear.shape[0]
        mat = np.eye(nstates) - alpha * self._linear.T
        return LUSolver(mat)

    def set_stepper(self, dt, method="rk2cn"):
        """Configure a timestepper for the semilinear model."""
        if method in SemiImplicit.methods():
            cls = SemiImplicit.lookup(method)
            self.stepper = cls(dt, self.linear, self.get_solver)

            def step(x):
                return self.stepper.step(x, self.nonlinear)
            self.step = step
            self.adjoint_stepper = cls(dt, self.adjoint,
                                       self.get_adjoint_solver)

            def adj_step(x, v):
                def f(w):
                    return self.adjoint_nonlinear(x, w)
                return self.adjoint_stepper.step(v, f)
            self.adjoint_step = adj_step
        else:
            super().set_stepper(dt, method=method)


class LUSolver:
    def __init__(self, mat):
        self.LU = lu_factor(mat)

    def __call__(self, rhs):
        return lu_solve(self.LU, rhs)


class BilinearModel(SemiLinearModel):
    """Model where the right-hand side is a bilinear function of the state

    Models have the form
        x' = c + L x + B(x, x)
    where B is bilinear
    """

    def __init__(self, c, L, B):
        self._affine = np.array(c)
        self._linear = np.array(L)
        self._bilinear = np.array(B)
        self.state_dim = self._linear.shape[0]

    def linear(self, x):
        return self._linear.dot(x)

    def adjoint(self, x):
        return self._linear.T.dot(x)

    def bilinear(self, a, b):
        """Evaluate the bilinear term B(a, b)"""
        return self._bilinear.dot(a).dot(b)

    def nonlinear(self, x):
        return self._affine + self.bilinear(x, x)

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
        n = len(V)
        if W is None:
            W = V
        assert len(W) == n

        # Let W1 = (W V')^{-1} W
        G = np.array([[np.dot(W[i], V[j]) for j in range(n)]
                      for i in range(n)])
        W1 = np.linalg.solve(G, W)
        # Now projection is given by P = V' W1, and W1 V' = Identity

        c = np.array([np.dot(W1[i], self._affine) for i in range(n)])
        L = np.array([[np.dot(W1[i], self.linear(V[j]))
                       for j in range(n)]
                      for i in range(n)])
        B = np.array([[[np.dot(W1[i], self.bilinear(V[j], V[k]))
                        for k in range(n)]
                       for j in range(n)]
                      for i in range(n)])
        return BilinearModel(c, L, B)


class LiftedROM(Model):
    """
    A reduced-order model that projects the dynamics onto linear subspaces

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
        n = len(V)
        if W is None:
            W = V
        assert len(W) == n

        # Let W1 = (W V')^{-1} W
        G = np.array([[np.dot(W[i], V[j]) for j in range(n)]
                      for i in range(n)])
        self.W1 = np.linalg.solve(G, W)
        # Now projection is given by P = V' W1, and W1 V' = Identity

    def rhs(self, z):
        fx = self.model.rhs(self.V.T @ z)
        return np.array([np.dot(self.W1[i], fx) for i in range(len(self.W1))])


class NetworkROM(Model):
    """
    A reduced-order model that projects onto the range of a romnet autoencoder

    Torch gradient information is not preserved

    The rom neural network is a differentiable idempotent operator
    P(x) = psid(psie(x)), where z = psie(x)

    The reduced-order model in state space is given by
    xdot = DP(x)f(x)

    The reduced-order model in the latent space is given by
    zdot = Dpsie(psid(z))f(psid(z))
    """

    def __init__(self, model, autoencoder):
        self.model = model
        self.autoencoder = autoencoder

    def rhs(self, z):
        with torch.no_grad():
            x = self.autoencoder.dec(z)
            _, v = self.autoencoder.d_enc(x, self.model.rhs(x))
            return v.numpy()


class GaussianProcessROM(Model):
    """
    A surrogate reduced-order model described by Gaussian process regression

    Let zdot = f(z) be the function we wish to approximate

    The Gaussian process reduced-order model is a function
    g(z) ~= f(z) = zdot  determined by Gaussian process regression
    """

    def __init__(self, inputdata, outputdata, kernel=None):
        if kernel is None:
            kernel = kernels.RBF()
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.gp.fit(inputdata, outputdata)

    def rhs(self, x):
        return self.gp.predict(x)
