#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import pytest

# for better compatibility with numpy arrays
torch.set_default_dtype(torch.float64)

class LayerPair(nn.Module):
    def __init__(self, dim_in, dim_out, angle=None):
        super().__init__()
        if angle is None:
            angle = np.pi / 8
        # activation function parameters
        self.a = 1. / np.sin(angle)**2 - 1. / np.cos(angle)**2
        self.b = 1. / np.sin(angle)**2 + 1. / np.cos(angle)**2
        self.d = self.b**2 - self.a**2

        self.dim_in = dim_in
        self.dim_out = dim_out

        # initialize weights
        Q = ortho_group.rvs(dim_in)[:,:dim_out]
        self.D = nn.Parameter(torch.tensor(Q))  # matrix for decoding
        self.X = nn.Parameter(torch.tensor(Q.transpose())) # matrix for encoding
        self.update()

        # initialize biases
        self.bias = nn.Parameter(-np.sqrt(2*self.a) / self.a *
            torch.matmul(self.D, torch.ones(self.dim_out,1)))

    def extra_repr(self):
        return "%d, %d" % (self.dim_in, self.dim_out)

    def update(self):
        self.E = (self.X @ self.D).inverse() @ self.X

    def enc_activ(self, x):
        """Activation function for encoder"""
        return (self.b * x - torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a

    def dec_activ(self, x):
        """Activation function for decoder"""
        return (self.b * x + torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a

    def d_enc_activ(self, x):
        return (self.b / self.a - self.d * x /
                (self.a * torch.sqrt( self.d * x**2 + 2*self.a )))

    def d_dec_activ(self, x):
        return (self.b / self.a + self.d * x /
                (self.a * torch.sqrt( self.d * x**2 + 2*self.a )))

    def enc(self, x):
        """Encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(self.E @ (x - self.bias)).squeeze(-1)

    def enc_slow(self, x):
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(
            torch.linalg.solve(torch.matmul(self.E, self.D),
                               torch.matmul(self.E, x - self.bias))).squeeze(-1)

    def dec(self, x):
        """Decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return (torch.matmul(self.D, self.dec_activ(x)) + self.bias).squeeze(-1)

    def forward(self, x):
        return self.dec(self.enc(x))

    def d_enc(self, x, v):
        """Tangent map of encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return (self.d_enc_activ(
                        torch.linalg.solve(
                            torch.matmul(self.E, self.D),
                            torch.matmul(self.E, x - self.bias )
                        )
                    ) * torch.linalg.solve(
                            torch.matmul(self.E, self.D),
                            torch.matmul(self.E, v)
                        )).squeeze(-1)

    def d_dec(self, x, v):
        """Tangent map of decoder"""
        x = torch.as_tensor(x)
        v = torch.as_tensor(v).unsqueeze(-1)
        return torch.matmul(self.D, self.d_dec_activ(x) * v ).squeeze(-1)

class ProjAE(nn.Module):
    """
    Autoencoder constrained to be a projection

    The autoencoder is built from a sequence of LayerPair objects
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList([
            LayerPair(self.dims[i], self.dims[i+1])
            for i in range(self.num_layers)
            ])

    def update(self):
        for pair in self.layers:
            pair.update()

    def enc(self, x):
        for i in range(self.num_layers):
            x = self.layers[i].enc(x)
        return x

    def dec(self, x):
        for i in reversed(range(self.num_layers)):
            x = self.layers[i].dec(x)
        return x

    def d_enc(self, x, v):
        for i in range(self.num_layers):
            v = self.layers[i].d_enc(x, v)
            x = self.layers[i].enc(x)
        return x, v

    def d_dec(self, x, v):
        for i in reversed(range(self.num_layers)):
            v = self.layers[i].d_dec(x, v)
            x = self.layers[i].dec(x)
        return x, v

    def d_autoenc(self, x, v):
        z, vz = self.d_enc(x, v)
        return self.d_dec(z, vz)

    def forward(self, x):
        return self.dec(self.enc(x))

def GAP_loss(Xpred, X, G):
    return torch.mean(torch.square(torch.sum(G * (Xpred - X), dim=1)))

def plot_pair():
    pair = LayerPair(2, 1)
    for p in pair.parameters():
        print(p)
    x = torch.tensor(np.linspace(-3,3,100))
    y = pair.enc_activ(x)
    z = pair.dec_activ(x)
    w = pair.dec_activ(y)
    err = np.max(np.abs((x-w).numpy()))
    print("Error = %g" % err)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.plot(x, w)
    plt.show()

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

    batch = np.array([x, 2*x, 3*x])
    yy = pair.enc(batch)
    assert yy.size() == torch.Size([3, 1])
    zz = pair.dec(yy)
    assert zz.size() == torch.Size([3, 2])
    zz = zz.detach().numpy()
    assert zz[0] == pytest.approx(z)
    assert zz[1] == pytest.approx(2*z)
    assert zz[2] == pytest.approx(3*z)

def test_pair22():
    pair = LayerPair(2, 2)
    x = np.ones(2)
    y = pair.enc(x)
    z = pair.dec(y).detach().numpy()
    assert z == pytest.approx(x)

    batch = np.array([x, 2*x, 3*x])
    ybatch = pair.enc(batch)
    zbatch = pair.dec(ybatch).detach().numpy()
    print(batch)
    print(zbatch)
    assert zbatch == pytest.approx(batch)

def test_pair_tangent():
    pairs = [LayerPair(2,1), LayerPair(2,2)]
    sizes = [1,2]
    x = np.ones(2)
    v0 = np.zeros(2)
    v1 = np.array([1.,0])
    v2 = np.array([0.,1])
    for i, pair in enumerate(pairs):
        w0 = pair.d_enc(x, v0)
        w1 = pair.d_enc(x, v1)
        w2 = pair.d_enc(x, v2)
        w12 = pair.d_enc(x, v1 + v2)
        assert w0.size() == torch.Size([sizes[i]])
        assert w1.size() == torch.Size([sizes[i]])
        w0 = w0.detach().numpy()
        assert w0 == pytest.approx(0)
        assert (w1.detach().numpy() + w2.detach().numpy() ==
                pytest.approx(w12.detach().numpy()))

        y = pair.enc(x)
        z0 = pair.d_dec(y, w0)
        z1 = pair.d_dec(y, w1)
        z2 = pair.d_dec(y, w2)
        z12 = pair.d_dec(y, w1 + w2)
        assert z0.size() == torch.Size([2])
        assert z1.size() == torch.Size([2])
        assert (z1.detach().numpy() + z2.detach().numpy() ==
                pytest.approx(z12.detach().numpy()))

        xbatch = np.array([x, x, x, x])
        vbatch = np.array([v0, v1, v2, v1+v2])
        wbatch = pair.d_enc(xbatch, vbatch)
        # zbatch = pair.d_dec(xbatch, wbatch)

def test_ae():
    dims = [5,3,2]
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

    batch = np.array([x, 2*x, 3*x])
    yy = ae.enc(batch)
    assert yy.size() == torch.Size([3, 2])
    zz = ae.dec(yy)
    assert zz.size() == torch.Size([3, 5])
    zz = zz.detach().numpy()
    assert zz[0] == pytest.approx(z)

if __name__ == "__main__":
    # plot_pair()
    # test_pair21()
    # test_pair2()
    test_ae()
