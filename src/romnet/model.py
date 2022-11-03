#!/usr/bin/env python
"""model - Define how a given state evolves in time."""

import abc

__all__ = ["timestepper", "Model"]


class Timestepper(abc.ABC):
    """Abstract base class for timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry = {}

    name = "default"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry[cls.name.lower()] = cls

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


class Euler(Timestepper):
    """Explicit Euler timestepper."""

    name = "euler"

    def step_(self, x, rhs):
        return x + self.dt * rhs(x)


class RK2(Timestepper):
    """Second-order Runge-Kutta timestepper."""

    name = "rk2"

    def step_(self, x, rhs):
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1)
        return x + (k1 + k2) / 2.


class RK4(Timestepper):
    """Fourth-order Runge-Kutta timestepper."""

    name = "rk4"

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
        self.stepper = timestepper(dt, self.rhs,
                                   adjoint_rhs=self.adjoint_rhs,
                                   method=method, **kwargs)
        self.step = self.stepper.step
        self.adjoint_step = self.stepper.adjoint_step
