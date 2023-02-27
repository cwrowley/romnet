from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ortho_group
from torch import Tensor, nn

from .typing import Vector, VectorField, TVector
from .model import Model

__all__ = ["ProjAE", "GAP_loss", "reduced_GAP_loss", "load_romnet", "save_romnet",
           "NetworkROM"]

# for better compatibility with numpy arrays
torch.set_default_dtype(torch.float64)


class LayerPair(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, angle: Optional[float] = None):
        super().__init__()
        if angle is None:
            angle = np.pi / 8
        # activation function parameters
        self.a = 1.0 / np.sin(angle) ** 2 - 1.0 / np.cos(angle) ** 2
        self.b = 1.0 / np.sin(angle) ** 2 + 1.0 / np.cos(angle) ** 2
        self.d = self.b**2 - self.a**2

        self.dim_in = dim_in
        self.dim_out = dim_out

        # initialize weights
        Q = ortho_group.rvs(dim_in)[:, :dim_out]
        self.D = nn.Parameter(torch.tensor(Q))  # decoding matrix
        self.X = nn.Parameter(torch.tensor(Q.transpose()))  # encoding matrix
        self.update()

        # initialize biases
        self.bias = nn.Parameter(
            -np.sqrt(2 * self.a) / self.a * (self.D @ torch.ones(self.dim_out, 1))
        )

    def extra_repr(self) -> str:
        return "%d, %d" % (self.dim_in, self.dim_out)

    def update(self) -> None:
        self.E = (self.X @ self.D).inverse() @ self.X

    def enc_activ(self, x: Tensor) -> Tensor:
        """Activation function for encoder"""
        return (self.b * x - torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a

    def dec_activ(self, x: Tensor) -> Tensor:
        """Activation function for decoder"""
        return (self.b * x + torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a

    def d_enc_activ(self, x: Tensor) -> Tensor:
        return self.b / self.a - self.d * x / (
            self.a * torch.sqrt(self.d * x**2 + 2 * self.a)
        )

    def d_dec_activ(self, x: Tensor) -> Tensor:
        return self.b / self.a + self.d * x / (
            self.a * torch.sqrt(self.d * x**2 + 2 * self.a)
        )

    def enc(self, x: TVector) -> Tensor:
        """Encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(self.E @ (x - self.bias)).squeeze(-1)

    def enc_slow(self, x: TVector) -> Tensor:
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(
            torch.linalg.solve(
                torch.matmul(self.X, self.D), torch.matmul(self.X, x - self.bias)
            )
        ).squeeze(-1)

    def dec(self, x: TVector) -> Tensor:
        """Decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return (torch.matmul(self.D, self.dec_activ(x)) + self.bias).squeeze(-1)

    def forward(self, x: TVector) -> Tensor:
        return self.dec(self.enc(x))

    def d_enc(self, x: TVector, v: TVector) -> Tensor:
        """Tangent map of encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return (self.d_enc_activ(self.E @ (x - self.bias)) * self.E @ v).squeeze(-1)

    def d_dec(self, x: TVector, v: TVector) -> Tensor:
        """Tangent map of decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return torch.matmul(self.D, self.d_dec_activ(x) * v).squeeze(-1)

    def regularizer(self) -> Tensor:
        P = self.D @ self.E
        return torch.log(torch.frobenius_norm(P) ** 2 / self.dim_out)


class ProjAE(nn.Module):
    """
    Autoencoder constrained to be a projection

    The autoencoder is built from a sequence of LayerPair objects
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList(
            [LayerPair(self.dims[i], self.dims[i + 1]) for i in range(self.num_layers)]
        )

    def update(self) -> None:
        for pair in self.layers:
            pair.update()

    def enc(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in self.layers:
            xout = layer.enc(xout)
        return xout

    def dec(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in reversed(self.layers):
            xout = layer.dec(xout)
        return xout

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in self.layers:
            vout = layer.d_enc(xout, vout)
            xout = layer.enc(xout)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in reversed(self.layers):
            vout = layer.d_dec(xout, vout)
            xout = layer.dec(xout)
        return xout, vout

    def d_autoenc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        z, vz = self.d_enc(x, v)
        return self.d_dec(z, vz)

    def forward(self, x: TVector) -> Tensor:
        return self.dec(self.enc(x))

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for layer in self.layers:
            total_regularizer += layer.regularizer()
        return total_regularizer

    def save(self, fname: str) -> None:
        torch.save(self, fname)


def load_romnet(fname: str) -> ProjAE:
    net = torch.load(fname)
    net.update()
    return net


def save_romnet(autoencoder: ProjAE, fname: str) -> None:
    torch.save(autoencoder, fname)


def GAP_loss(X_pred: Tensor, X: Tensor, G: Tensor) -> Tensor:
    return torch.mean(torch.square(torch.sum(G * (X_pred - X), dim=1)))


def reduced_GAP_loss(X_pred: Tensor, X: Tensor, G: Tensor, XdotG: Tensor) -> Tensor:
    return torch.mean(torch.square(XdotG - torch.sum(G * X_pred, dim=1)))


class NetworkROM(Model):
    """
    Return a reduced-order model that projects the dynamics onto the range of
    a romnet autoencoder.

    The romnet autoencoder is a a differentiable idempotent operator

    .. math:: \\tilde x = P(x) = \\psi_d(\\psi_e(x)),

    where

    .. math:: z = \\psi_e(x), \\quad \\tilde x = \\psi_d(z)

    are the encoder and decoder, respectively. The reduced-order model in
    state space is given by

    .. math:: \\dot{\\tilde{x}} = \\mathrm{D}_x P(\\tilde x) f(\\tilde x),

    and in the latent space by

    .. math:: \\dot z = h(z) = \\mathrm{D}_x \\psi_e(\\psi_d(z)) f(\\psi_d(z)),

    where

    .. math:: \\dot x = f(x)

    is the full-order model.

    Attributes:
        _rhs (VectorField): Right hand side of the full-order model.
        autoencoder (ProjAE): Romnet autoencoder used in reduced-order model.

    Note:
        Torch gradient information is not preserved.
    """

    def __init__(self, rhs: VectorField, autoencoder: "ProjAE") -> None:
        self._rhs = rhs
        self.autoencoder = autoencoder

    def rhs(self, z: TVector) -> Vector:
        """Return the right-hand-side of the reduced-order model ODE in the
        latent space, z' = h(z).
        """
        with torch.no_grad():
            x = self.autoencoder.dec(z)
            _, v = self.autoencoder.d_enc(x, self._rhs(x))
            return v.numpy()

    def adjoint_rhs(self, x: Vector, v: Vector) -> Vector:
        """For the right-hand-side of the reduced-order model ODE in the
        latent space h(z), return Dh(z)^T v.
        """
        return super().adjoint_rhs(x, v)
