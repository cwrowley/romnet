"""model - Define how a given state evolves in time."""

import abc

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

from .timestepper import SemiImplicit, Timestepper

__all__ = ["Model", "SemiLinearModel", "BilinearModel", "LUSolver"]


class Model:
    """
    Class defining an ODE dx/dt = f(x)

    The constructor requires we input the right-hand side of the ODE, x' = f(x)
    The right-hand side method is called rhs(x)
    """

    def __init__(self, rhs, adjoint_rhs=None, output=None,
                 adjoint_output=None):
        setattr(self, "rhs", rhs)
        if adjoint_rhs is not None:
            setattr(self, "adjoint_rhs", adjoint_rhs)
        if adjoint_rhs is not None:
            setattr(self, "output", output)
        if adjoint_rhs is not None:
            setattr(self, "adjoint_output", adjoint_output)

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

    def project(self, V, W=None):
        """
        Returns a reduced-order model that projects onto linear subspaces

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

        def rom_rhs(z):
            x = sum((mode * c for mode, c in zip(V, z)))
            fx = self.rhs(x)
            return np.array([np.dot(W1[i], fx) for i in range(len(W1))])

        return Model(rom_rhs)


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
    """A class for solving linear systems A x = b

    Args:
        mat(array): the matrix A

    When instantiated, an LU factorization of A is computed, and this is
    used when solving the system for a given right-hand side b.
    """

    def __init__(self, mat):
        self.LU = lu_factor(mat)

    def __call__(self, rhs):
        """Solve the system A x = rhs for x

        Args:
            rhs(array): the right-hand side of the equation to be solved

        Returns:
            The solution x
        """
        return lu_solve(self.LU, rhs)


class BilinearModel(SemiLinearModel):
    """Model where the right-hand side is a bilinear function of the state

    Models have the form

        x' = c + L x + B(x, x)

    where B is bilinear

    Args:
        c(array_like): vector containing the constant terms c
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
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
        """Return a reduced-order model by projecting onto linear subspaces

        Rows of V determine the subspace to project onto, and
        rows of W determine the direction of projection

        That is, the projection is given by

        .. math:: V^T (WV^T)^{-1} W

        The number of states in the reduced-order model is the number of rows
        in V (or W).

        Args:
            V(list): List of modes to project onto.
                The model is projected onto a subspace spanned by the elements
                of this list.
            W(list): List of adjoint modes.
                The nullspace of the projection is the orthogonal complement
                of the subspace spanned by the elements of W.
                If not specified, an orthogonal projection is used (W = V)

        Returns:
            A :class:`BilinearModel` containing the desired projection
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
