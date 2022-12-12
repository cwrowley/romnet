import numpy as np
from romnet.model import Model
from romnet.model import BilinearModel
from romnet.model import NetworkLiftedROM
from romnet.model import LinearLiftedROM
from romnet.autoencoder import ProjAE
import pytest


@pytest.fixture
def bilinear_model():
    # simple bilinear model:
    #   x' = y + xy
    #   y' = 2 - x
    c = np.array([0, 2])
    L = np.array([[0, 1], [-1, 0]])
    B = np.zeros((2, 2, 2))
    B[0, 0, 1] = 1
    return BilinearModel(c, L, B)


def test_rhs_bilinear(bilinear_model):
    x = np.random.randn(2)
    xdot = np.array([x[1] + x[0] * x[1], 2 - x[0]])
    assert xdot == pytest.approx(bilinear_model.rhs(x))


def test_adjoint_bilinear(bilinear_model):
    x = np.random.randn(2)
    v = np.random.randn(2)
    w = np.random.randn(2)
    adj = bilinear_model.adjoint_rhs(x, w)
    L = np.array([[x[1], 1 + x[0]], [-1, 0]])
    assert L.dot(v).dot(w) == pytest.approx(adj.dot(v))


@pytest.fixture
def beta():
    return 20


@pytest.fixture
def example1(beta):
    """
    Example bilinear system from Sam's CoBRAS paper:

    x' = -x + beta x z
    y' = -2y + beta y z
    z' = -5z
    """
    c = np.zeros(3)
    L = np.diag([-1, -2, -5])
    B = np.zeros((3, 3, 3))
    B[0, 0, 2] = beta  # x z term in the first equation
    B[1, 1, 2] = beta  # x y term in the second equation
    return BilinearModel(c, L, B)


class BilinearExample(BilinearModel):
    """
    Example bilinear system from Sam's CoBRAS paper:

    x' = -x + beta x z
    y' = -2y + beta y z
    z' = -5z

    this time implemented as a class
    """
    def __init__(self, beta):
        self.beta = beta
        self.affine = np.zeros(3)
        self.linear = np.diag([-1, -2, -5])

    def bilinear(self, a, b):
        result = np.zeros(3)
        result[0] = a[0] * b[2]
        result[1] = a[1] * b[2]
        return self.beta * result

    def adjoint_nonlinear(self, x, v):
        result = np.zeros(3)
        result[0] = x[2] * v[0]
        result[1] = x[2] * v[1]
        result[2] = x[0] * v[0] + x[1] * v[1]
        return self.beta * result


@pytest.fixture
def example2(beta):
    return BilinearExample(beta)


def test_examples_agree(example1, example2):
    x = np.random.randn(3)
    v = np.random.randn(3)
    assert example1.rhs(x) == pytest.approx(example2.rhs(x))
    assert (example1.adjoint_nonlinear(x, v) ==
            pytest.approx(example2.adjoint_nonlinear(x, v)))


def test_project_xy(example1):
    """Project the example orthogonally onto the x-y plane"""
    V = np.array([[1, 0, 0], [0, 1, 0]])
    rom = example1.project(V)
    assert rom.affine == pytest.approx(np.zeros(2))
    assert rom.linear == pytest.approx(np.diag([-1, -2]))
    z = np.random.randn(2)
    assert rom.nonlinear(z) == pytest.approx(0)


def compare_projection(full_model):
    """Oblique projection"""
    eps = 0.1
    V = np.array([[1 - eps, 0, eps],
                  [0, 1 - eps, eps]])
    W = np.array([[1, 0, 1],
                  [0, 1, 1]])
    W1 = np.array([[1, -eps, 1 - eps],
                   [-eps, 1, 1 - eps]]) / (1 - eps**2)
    rom1 = full_model.project(V, W)
    rom2 = full_model.project(V, W1)
    z = np.random.randn(2)
    x = V.T @ z
    proj_rhs = W1 @ full_model.rhs(x)
    assert rom1.rhs(z) == pytest.approx(proj_rhs)
    assert rom2.rhs(z) == pytest.approx(proj_rhs)


def test_projection1(example1):
    compare_projection(example1)


def test_projection2(example2):
    compare_projection(example2)


class Pitchfork(Model):
    """
    Example pitchfork bifurcation model in two dimensions.

    xdot = x(a^2 - x^2)
    ydot = lam * (x^2 - y)

    Two asymptotically stable fixed points at (-a, a^2), (a, a^2)
    One unstable fixed point at (0,0)

    a controls the fix point locations
    lam controls the speed of fast dynamics

    Slow manifold near critical manifold of {(x,y) : y = x^2}
    """

    def __init__(self, a=1., lam=10.):
        self.a = a
        self.lam = lam

    def rhs(self, x):
        xdot = x[0] * (self.a**2 - x[0]**2)
        ydot = self.lam * (x[0]**2 - x[1])
        return np.array([xdot, ydot])

    def jac(self, x):
        df1 = [self.a**2 - 3 * x[0]**2, 0]
        df2 = [2 * self.lam * x[0], -self.lam]
        return np.array([df1, df2])

    def adjoint_rhs(self, x, v):
        return self.jac(x).T @ v


@pytest.fixture
def pitchfork_model():
    return Pitchfork()


@pytest.fixture
def pitchfork_autoencoder():
    """
    Return an untrained romnet autoencoder with latent dimension
    equal to the state dimension.

    The weights will be initialized to the identity as is written
    in the ProjAE class in autoencoder.py

    Thus, P = Id and DP = Id on R^2.
    """
    dims = [2, 2]
    return ProjAE(dims)


def test_pitchfork_autoencoder(pitchfork_autoencoder):
    x = np.random.randn(2)
    v = np.random.randn(2)
    assert x == pytest.approx(pitchfork_autoencoder(x).detach().numpy())
    _, dP = pitchfork_autoencoder.d_autoenc(x, v)
    assert v == pytest.approx(dP.detach().numpy())


@pytest.fixture
def pitchfork_network_model(pitchfork_model, pitchfork_autoencoder):
    return NetworkLiftedROM(pitchfork_model, pitchfork_autoencoder)


def test_pitchfork_network_lifted_model(pitchfork_model, pitchfork_autoencoder,
                                        pitchfork_network_model):
    x = np.random.randn(2)
    v1 = pitchfork_model.rhs(x)
    z = pitchfork_autoencoder.enc(x)
    vv = pitchfork_network_model.rhs(z)
    _, v2 = pitchfork_autoencoder.d_dec(z, vv)
    assert v1 == pytest.approx(v2.detach().numpy())


def test_pitchfork_linear_lifted_model(pitchfork_model):
    Phi_T = np.array([[1, 0]])
    Psi_T = np.array([[1, 0]])
    rom = LinearLiftedROM(pitchfork_model, Phi_T, Psi_T)
    x = np.random.randn(2)
    z = Psi_T @ x
    direct = Psi_T @ pitchfork_model.rhs(Phi_T.T @ z)
    romClass = rom.rhs(z)
    assert direct == pytest.approx(romClass)
