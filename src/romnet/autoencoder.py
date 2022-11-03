import torch
from torch import nn
import numpy as np
from scipy.stats import ortho_group

__all__ = ["ProjAE", "GAP_loss"]

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
        Q = ortho_group.rvs(dim_in)[:, :dim_out]
        self.D = nn.Parameter(torch.tensor(Q))  # decoding matrix
        self.X = nn.Parameter(torch.tensor(Q.transpose()))  # encoding matrix
        self.update()

        # initialize biases
        self.bias = nn.Parameter(-np.sqrt(2*self.a) / self.a *
                                 torch.matmul(self.D,
                                              torch.ones(self.dim_out, 1)))

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
                (self.a * torch.sqrt(self.d * x**2 + 2*self.a)))

    def d_dec_activ(self, x):
        return (self.b / self.a + self.d * x /
                (self.a * torch.sqrt(self.d * x**2 + 2*self.a)))

    def enc(self, x):
        """Encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(self.E @ (x - self.bias)).squeeze(-1)

    def enc_slow(self, x):
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(
            torch.linalg.solve(torch.matmul(self.E, self.D),
                               torch.matmul(self.E,
                                            x - self.bias))).squeeze(-1)

    def dec(self, x):
        """Decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return (torch.matmul(self.D, self.dec_activ(x))
                + self.bias).squeeze(-1)

    def forward(self, x):
        return self.dec(self.enc(x))

    def d_enc(self, x, v):
        """Tangent map of encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return (self.d_enc_activ(
                        torch.linalg.solve(
                            torch.matmul(self.E, self.D),
                            torch.matmul(self.E, x - self.bias)
                        )
                    ) * torch.linalg.solve(
                            torch.matmul(self.E, self.D),
                            torch.matmul(self.E, v)
                        )).squeeze(-1)

    def d_dec(self, x, v):
        """Tangent map of decoder"""
        x = torch.as_tensor(x)
        v = torch.as_tensor(v).unsqueeze(-1)
        return torch.matmul(self.D, self.d_dec_activ(x) * v).squeeze(-1)


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
        for layer in self.layers:
            x = layer.enc(x)
        return x

    def dec(self, x):
        for layer in reversed(self.layers):
            x = layer.dec(x)
        return x

    def d_enc(self, x, v):
        for layer in self.layers:
            v = layer.d_enc(x, v)
            x = layer.enc(x)
        return x, v

    def d_dec(self, x, v):
        for layer in reversed(self.layers):
            v = layer.d_dec(x, v)
            x = layer.dec(x)
        return x, v

    def d_autoenc(self, x, v):
        z, vz = self.d_enc(x, v)
        return self.d_dec(z, vz)

    def forward(self, x):
        return self.dec(self.enc(x))


def GAP_loss(Xpred, X, G):
    return torch.mean(torch.square(torch.sum(G * (Xpred - X), dim=1)))
