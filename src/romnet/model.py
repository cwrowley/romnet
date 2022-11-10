#!/usr/bin/env python
"""model - Define how a given state evolves in time."""

import abc
import numpy as np
from scipy.linalg import lu_factor, lu_solve

__all__ = ["timestepper", "Model", "SemiLinearModel"]


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

    def __init__(self, dt, linear, nonlinear, adjoint_nonlinear):
        self._dt = dt
        self.linear = linear
        self.nonlinear = nonlinear
        self.adjoint_nonlinear = adjoint_nonlinear
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
    def step(self, x):
        """Advance the state x by one step."""

    @abc.abstractmethod
    def adjoint_step(self, x, v):
        """Advance the adjoint variable v by one step, linearized about x."""

    @classmethod
    def methods(cls):
        return list(cls.__registry.keys())


class RK2CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 2nd-order Runge-Kutta."""

    def update(self):
        nstates = self.linear.shape[0]
        mat = np.eye(nstates) - 0.5 * self.dt * self.linear
        self.LU = lu_factor(mat)

    def step(self, x):
        rhs_linear = x + 0.5 * self._dt * np.dot(self.linear, x)
        Nx = self.nonlinear(x)
        rhs1 = rhs_linear + self._dt * Nx
        x1 = lu_solve(self.LU, rhs1)
        rhs2 = rhs_linear + 0.5 * self._dt * (Nx + self.nonlinear(x1))
        x_next = lu_solve(self.LU, rhs2)
        return x_next

    def adjoint_step(self, x):
        raise NotImplementedError("Adjoint step not yet implemented")


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


def timestepper(dt, rhs, method="rk2", **kwargs):
    """Return a timestepper corresponding to the given `method`."""
    cls = Timestepper.lookup(method)
    return cls(dt, rhs, **kwargs)


class Model(abc.ABC):
    """Abstract base class defining an ODE dx/dt = f(x).

    Subclasses must override two methods:
      rhs(x) - returns the right-hand side f(x)
      adjoint_rhs(x, v) -
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
        self.stepper = cls(dt, self.rhs, adjoint_rhs=self.adjoint_rhs, **kwargs)
        self.step = self.stepper.step
        self.adjoint_step = self.stepper.adjoint_step


class SemiLinearModel(Model):
    """Abstract base class for semi-linear models of the form x' = A x + N(x)."""

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
