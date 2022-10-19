import torch

__all__ = ["train_loop", "test_loop"]


def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, G) in enumerate(dataloader):
        Xpred = autoencoder(X)
        loss = loss_fn(Xpred, X, G)

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
        for X, G in dataloader:
            Xpred = autoencoder(X)
            loss += loss_fn(Xpred, X, G).item()

    loss /= num_batches
    print(f"Average loss: {loss:>7f}")
