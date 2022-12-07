import numpy as np
from romnet.model import BilinearModel
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
