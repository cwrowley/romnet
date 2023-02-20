#!/usr/bin/env python

import torch
import numpy as np
import pytest
from romnet.autoencoder import LayerPair, ProjAE


def test_pair21():
    pair = LayerPair(2, 1)
    x = np.ones(2)
    y = pair.enc(x)
    assert y.size() == torch.Size([1])
    z = pair.dec(y)
    assert z.size() == torch.Size([2])
    z2 = pair.forward(z)
    z = z.detach().numpy()
    z2 = z2.detach().numpy()
    assert z2 == pytest.approx(z)

    batch = np.array([x, 2 * x, 3 * x])
    yy = pair.enc(batch)
    assert yy.size() == torch.Size([3, 1])
    zz = pair.dec(yy)
    assert zz.size() == torch.Size([3, 2])
    zz = zz.detach().numpy()
    assert zz[0] == pytest.approx(z)
    assert zz[1] == pytest.approx(2 * z)
    assert zz[2] == pytest.approx(3 * z)


def test_pair22():
    pair = LayerPair(2, 2)
    x = np.ones(2)
    y = pair.enc(x)
    z = pair.dec(y).detach().numpy()
    assert z == pytest.approx(x)

    batch = np.array([x, 2 * x, 3 * x])
    ybatch = pair.enc(batch)
    zbatch = pair.dec(ybatch).detach().numpy()
    print(batch)
    print(zbatch)
    assert zbatch == pytest.approx(batch)


def test_pair_tangent():
    pairs = [LayerPair(2, 1), LayerPair(2, 2)]
    sizes = [1, 2]
    x = np.ones(2)
    v0 = np.zeros(2)
    v1 = np.array([1.0, 0])
    v2 = np.array([0.0, 1])
    for i, pair in enumerate(pairs):
        w0 = pair.d_enc(x, v0)
        w1 = pair.d_enc(x, v1)
        w2 = pair.d_enc(x, v2)
        w12 = pair.d_enc(x, v1 + v2)
        assert w0.size() == torch.Size([sizes[i]])
        assert w1.size() == torch.Size([sizes[i]])
        assert w0.detach() == pytest.approx(0)
        assert w1.detach() + w2.detach() == pytest.approx(w12.detach())

        y = pair.enc(x)
        z0 = pair.d_dec(y, w0)
        z1 = pair.d_dec(y, w1)
        z2 = pair.d_dec(y, w2)
        z12 = pair.d_dec(y, w1 + w2)
        assert z0.size() == torch.Size([2])
        assert z1.size() == torch.Size([2])
        assert z1.detach() + z2.detach() == pytest.approx(z12.detach())

        xbatch = np.array([x, x, x, x])
        vbatch = np.array([v0, v1, v2, v1 + v2])
        wbatch = pair.d_enc(xbatch, vbatch)
        assert wbatch.size() == torch.Size([4, sizes[i]])
        for w, ww in zip(wbatch.detach(), [w0, w1, w2, w12]):
            assert w == pytest.approx(ww.detach())

        y = y.detach().numpy()
        ybatch = np.array([y, y, y, y])
        zbatch = pair.d_dec(ybatch, wbatch)
        for z, zz in zip(zbatch.detach(), [z0, z1, z2, z12]):
            assert z == pytest.approx(zz.detach())


def test_ae():
    dims = [5, 3, 2]
    ae = ProjAE(dims)
    x = np.ones(5)
    y = ae.enc(x)
    assert y.size() == torch.Size([2])
    z = ae.dec(y)
    assert z.size() == torch.Size([5])

    z2 = ae.forward(z)
    z2 = z2.detach().numpy()
    z = z.detach().numpy()
    assert z2 == pytest.approx(z)

    batch = np.array([x, 2 * x, 3 * x])
    yy = ae.enc(batch)
    assert yy.size() == torch.Size([3, 2])
    zz = ae.dec(yy)
    assert zz.size() == torch.Size([3, 5])
    zz = zz.detach().numpy()
    assert zz[0] == pytest.approx(z)


if __name__ == "__main__":
    test_ae()
    test_pair_tangent()
