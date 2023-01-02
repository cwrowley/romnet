import torch
import numpy as np
import pickle

__all__ = ["train_loop", "test_loop", "CoBRAS"]


def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data_tuple in enumerate(dataloader):
        Xpred = autoencoder(data_tuple[0])
        loss = loss_fn(Xpred, data_tuple)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        autoencoder.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(data_tuple[0])
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    loss = 0

    with torch.no_grad():
        for data_tuple in dataloader:
            Xpred = autoencoder(data_tuple[0])
            loss += loss_fn(Xpred, data_tuple).item()

    loss /= num_batches
    print(f"Average loss: {loss:>7f}")


class CoBRASReducedGradientDataset:
    def __init__(self, Xi, Gam, diagYXT):
        self.Xi = np.array(Xi)
        self.Gam = np.array(Gam)
        self.diagYXT = np.array(diagYXT)

    def __len__(self):
        return self.Xi.shape[0]

    def __getitem__(self, i):
        return self.Xi[i], self.Gam[i], self.diagYXT[i]

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class CoBRAS:
    def __init__(self, X, Y, mode_num):
        self.mode_num = mode_num
        self.update_svd(X, Y)
        self.update_projectors(X, Y, mode_num)

    def update_svd(self, X, Y):
        self.U, self.s, self.VH = np.linalg.svd(np.dot(Y, X.T),
                                                full_matrices=False,
                                                compute_uv=True)

    def update_projectors(self, X, Y, mode_num):
        self.Phi = np.dot(X.T,
                          self.VH[:mode_num, :].T) / np.sqrt(self.s[:mode_num])
        self.Psi = np.dot(Y.T,
                          self.U[:, :mode_num]) / np.sqrt(self.s[:mode_num])

    def projectors(self):
        return self.Phi.T, self.Psi.T

    def save_projectors(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump((self.Phi, self.Psi), fp, pickle.HIGHEST_PROTOCOL)

    def generate_gradient_dataset(self, X, Y):
        return CoBRASReducedGradientDataset(X @ self.Psi,
                                            Y @ self.Phi,
                                            np.diag(np.dot(Y, X.T)))
