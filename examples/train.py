#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import romnet


def train_autoencoder(basename: str):
    # load data
    print(f"Loading data from {basename}_train.dat")
    training_data = romnet.load(basename + "_train.dat")
    test_data = romnet.load(basename + "_test.dat")

    # autoencoder hyperparameters by example
    #   noack
    learning_rate = 1.0e-4
    batch_size = 400
    num_epochs = 750
    dims = [3, 3, 3, 3, 3, 3, 2]
    gamma1 = 1.0e-4
    autoencoder1 = romnet.ProjAE(dims)
    num_linear_layers = 3
    gamma2 = 1.0e-4
    autoencoder2 = romnet.MultiLinear(dims[-1], num_linear_layers)
    autoencoder = romnet.AEList([autoencoder1, autoencoder2])

    # save initial weights
    romnet.save_romnet(autoencoder, basename + "_initial" + ".romnet")

    # load autoencoder
    # autoencoder = romnet.load_romnet(basename + ".romnet")

    # loss function
    def loss_fn(X_pred, X, G):
        loss = romnet.GAP_loss(X_pred, X, G)
        reg = (gamma1 * autoencoder.regularizer_component(0)
               + gamma2 * autoencoder.regularizer_component(1))
        return loss + reg

    # train autoencoder
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_epoch = []
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-----------------")
        romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
        loss = romnet.test_loop(test_dataloader, autoencoder, loss_fn)
        loss_epoch.append(loss)

    # plot average test loss
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_epoch)
    ax.set_ylabel("Average Test Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    fig.savefig(basename + "_loss.pdf", format="pdf")

    # save autoencoder
    romnet.save_romnet(autoencoder, basename + ".romnet")

    print("Done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train_autoencoder("noack")
    elif len(sys.argv) == 2:
        train_autoencoder(sys.argv[1])
