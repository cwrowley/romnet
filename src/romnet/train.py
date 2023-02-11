import torch
import numpy as np
import pickle

__all__ = ["train_loop", "test_loop", "CoBRAS"]


def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data_tuple in enumerate(dataloader):
        X = data_tuple[0]
        Xpred = autoencoder(X)
        loss = loss_fn(Xpred, *data_tuple)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        autoencoder.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    loss = 0

    with torch.no_grad():
        for data_tuple in dataloader:
            X = data_tuple[0]
            Xpred = autoencoder(X)
            loss += loss_fn(Xpred, *data_tuple).item()

    loss /= num_batches
    print(f"Average loss: {loss:>7f}")


class ProjectedGradientDataset:
    def __init__(self, X, G, XdotG):
        self.X = np.array(X)
        self.G = np.array(G)
        self.XdotG = np.array(XdotG)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.G[i], self.XdotG[i]

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class CoBRAS:
    def __init__(self, X, Y):
        self.U, self.s, self.VH = np.linalg.svd(np.dot(Y, X.T),
                                                full_matrices=False,
                                                compute_uv=True)
        self.Phi = np.dot(X.T, self.VH.T) / np.sqrt(self.s)
        self.Psi = np.dot(Y.T, self.U) / np.sqrt(self.s)

    def projectors(self):
        return self.Phi.T, self.Psi.T

    def save_projectors(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump((self.Phi, self.Psi), fp, pickle.HIGHEST_PROTOCOL)

    def project(self, X, G, rank):
        XdotG = np.array([np.dot(x, g) for x, g in zip(X, G)])
        Xproj = X @ self.Psi[:, :rank]
        Gproj = G @ self.Phi[:, :rank]
        return ProjectedGradientDataset(Xproj, Gproj, XdotG)
