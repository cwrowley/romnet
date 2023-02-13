#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import sys
import romnet


def train_autoencoder(basename):
    # load data
    print(f"Loading data from {basename}_train.dat")
    training_data = romnet.load(basename + "_train.dat")
    # test_data = romnet.load(basename + "_test.dat")

    # autoencoder hyperparameters by example
    #   noack
    learning_rate = 1.e-3
    batch_size = 400
    num_epochs = 10  # 500
    dims = [3, 3, 3, 3, 3, 2]
    autoencoder = romnet.ProjAE(dims)
    # gamma = 0
    #   cgl
    """
    learning_rate = 1.e-3
    batch_size = 64
    num_epochs = 50
    dims = [15, 15, 15, 15, 15, 14]
    autoencoder = romnet.ProjAE(dims)
    gamma = 1.e-3
    """

    # load autoencoder
    # autoencoder = romnet.load_romnet(basename + ".romnet")

    # loss function
    def loss_fn(X_pred, X, G):
        loss = romnet.GAP_loss(X_pred, X, G)
        # reg = gamma * autoencoder.regularizer()
        return loss  # + reg
    """
    def loss_fn(Xi_pred, X, G, XdotG):
        loss = romnet.reduced_GAP_loss(Xi_pred, X, G, XdotG)
        reg = gamma * autoencoder.regularizer()
        return loss + reg
    """

    # train autoencoder
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data,
    #                              batch_size=batch_size, shuffle=True)

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-----------------")
        romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
        # romnet.test_loop(test_dataloader, autoencoder, loss_fn)

    # save autoencoder
    romnet.save_romnet(autoencoder, basename + ".romnet")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train_autoencoder("noack")
    else:
        train_autoencoder(sys.argv[1])
