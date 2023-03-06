#!/usr/bin/env python

import sys

import romnet
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train_autoencoder(basename, train_num="", savefig=False):
    # load data
    print(f"Loading data from {basename}_train.dat")
    training_data = romnet.load(basename + "_train.dat")
    test_data = romnet.load(basename + "_test.dat")

    # autoencoder hyperparameters by example
    #   1. noack
    """
    learning_rate = 1.e-4
    batch_size = 400
    num_epochs = 1000
    dims = [3, 3, 3, 3, 3, 3, 3, 2]
    autoencoder = romnet.ProjAE(dims)
    # gamma = 0
    """
    #   2. cgl
    learning_rate = 1.e-4
    batch_size = 512
    num_epochs = 200
    dims = [15, 12, 12, 2]
    autoencoder = romnet.ProjAE(dims)
    # gamma = 0

    # save initial weights
    romnet.save_romnet(autoencoder, basename + train_num + "_initial" + ".romnet")

    # load autoencoder
    # autoencoder = romnet.load_romnet(basename + train_num + ".romnet")

    # loss function
    #   1. GAP loss
    """
    def loss_fn(X_pred, X, G):
        loss = romnet.GAP_loss(X_pred, X, G)
        # reg = gamma * autoencoder.regularizer()
        return loss  # + reg
    """
    #   2. reduced GAP loss
    def loss_fn(X_pred, X, G, XdotG):
        loss = romnet.reduced_GAP_loss(X_pred, X, G, XdotG)
        # reg = gamma * autoencoder.regularizer()
        return loss  # + reg

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
    ax.set_xlabel("Epochs")
    if savefig:
        fig.savefig(basename + train_num + "_loss.pdf", format="pdf")

    # save autoencoder
    romnet.save_romnet(autoencoder, basename + train_num + ".romnet")

    print("Done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train_autoencoder("noack")
    elif len(sys.argv) == 2:
        train_autoencoder(sys.argv[1])
    elif len(sys.argv) == 3:
        train_autoencoder(sys.argv[1], "_" + sys.argv[2])
    elif (len(sys.argv) == 4) and (sys.argv[3] == "savefig"):
        train_autoencoder(sys.argv[1], "_" + sys.argv[2], savefig=True)
