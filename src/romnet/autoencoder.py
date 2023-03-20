from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import ortho_group
from torch import Tensor, nn
from abc import ABC, abstractmethod

from .typing import Vector

__all__ = ["ProjAE", "GAP_loss", "reduced_GAP_loss", "load_romnet", "save_romnet",
           "recon_loss", "reduced_recon_loss"]

# for better compatibility with numpy arrays
torch.set_default_dtype(torch.float64)

TVector = Union[Vector, Tensor]


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


class AE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dim_in(self) -> int:
        """Input dimension"""

    @property
    @abstractmethod
    def dim_out(self) -> int:
        """Output dimension"""

    def update(self) -> None:
        """
        Update model parameters pre- and post-optimization step.

        Default set to null function.
        """
        pass

    @abstractmethod
    def enc(self, x: TVector) -> Tensor:
        """Encoder"""

    @abstractmethod
    def dec(self, x: TVector) -> Tensor:
        """Decoder"""

    def forward(self, x: TVector) -> Tensor:
        """Decoder composed with encoder"""
        return self.dec(self.enc(x))

    @abstractmethod
    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of encoder"""

    @abstractmethod
    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of decoder"""

    def d_autoenc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of decoder composed with encoder"""
        z, vz = self.d_enc(x, v)
        return self.d_dec(z, vz)

    def regularizer(self) -> float:
        """Total regularizer"""
        raise NotImplementedError(
            "Regularizer not implemented for class %s" % self.__class__.__name__
        )

    def save(self, fname: str) -> None:
        """Save autoencoder"""
        torch.save(self, fname)


class ProjAE(AE):
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

    def dim_in(self) -> int:
        return self.dims[0]

    def dim_out(self) -> int:
        return self.dims[-1]

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

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for layer in self.layers:
            total_regularizer += layer.regularizer()
        return total_regularizer


class MultiLinear(AE):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.dim = dim
        self.layers = [
            nn.Parameter(torch.tensor(ortho_group.rvs(dim)))
            for _ in range(num_layers)
        ]
        self.update()

    def dim_in(self) -> int:
        return self.dim

    def dim_out(self) -> int:
        return self.dim

    def update(self) -> None:
        self.E = torch.eye(self.dim)
        self.D = torch.eye(self.dim)
        for layer in self.layers:
            self.E = layer @ self.E
        for layer in reversed(self.layers):
            self.D = layer.inverse() @ self.D

    def enc(self, x: TVector) -> Tensor:
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.E @ x

    def dec(self, x: TVector) -> Tensor:
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.D @ x

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        v = torch.as_tensor(v).unsqueeze(-1)
        return self.enc(x), self.E @ v

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        v = torch.as_tensor(v).unsqueeze(-1)
        return self.dec(x), self.D @ v

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for layer in self.layers:
            total_regularizer += torch.sum(layer * layer).item()
        return total_regularizer


class AEList(AE):
    def __init__(self, ae_list: List[AE], num_linear: int):
        super().__init__()
        self.ae_list = ae_list
        self.num_ae = len(ae_list)
        if self.num_ae < 2:
            raise ValueError(
                "projae_list needs length >=2, current length is {}".format(self.num_ae)
            )
        for i in range(self.num_ae-1):
            dim_out = ae_list[i].dim_out
            dim_in = ae_list[i + 1].dim_in
            if dim_out != dim_in:
                raise ValueError(
                    "Element {} of ae_list has dim_out = {} and next element has dim_in = {}".format(i, dim_out, dim_in)
                )

    def dim_in(self) -> int:
        return self.ae_list[0].dim_in

    def dim_out(self) -> int:
        return self.ae_list[-1].dim_out

    def update(self) -> None:
        for ae in self.ae_list:
            ae.update()

    def enc(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for ae in self.ae_list:
            xout = ae.enc(xout)
        return xout

    def dec(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for ae in reversed(self.ae_list):
            xout = ae.dec(xout)
        return xout

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for ae in self.ae_list:
            xout, vout = ae.d_enc(xout, vout)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for ae in self.ae_list:
            xout, vout = ae.d_dec(xout, vout)
        return xout, vout

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for ae in self.ae_list:
            total_regularizer += ae.regularizer()
        return total_regularizer

    def regularizer_component(self, ae_idx: int) -> float:
        return self.ae_list[ae_idx].regularizer()


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


def recon_loss(X_pred: Tensor, X: Tensor) -> Tensor:
    E = X - X_pred
    return torch.mean(torch.sum(E * E, dim=1))


def reduced_recon_loss(
    X_pred: Tensor,
    X: Tensor,
    G: Tensor,
    XdotX: Tensor,
    M: Tensor
) -> Tensor:
    term1 = - 2 * torch.sum(G * X_pred, dim=1)
    term2 = torch.sum(X_pred * (X_pred @ M.T), dim=1)
    return torch.mean(XdotX + term1 + term2)
