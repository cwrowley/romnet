#!/usr/bin/env python

import sys

import romnet
import torch
from torch.utils.data import DataLoader


def train_autoencoder(basename, train_num=""):
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
    autoencoder = romnet.ProjAE(dims)
    # gamma = 0

    # save initial weights
    romnet.save_romnet(autoencoder, basename + train_num + "_initial" + ".romnet")

    # load autoencoder
    # autoencoder = romnet.load_romnet(basename + train_num + ".romnet")

    # loss function
    def loss_fn(X_pred, X, G):
        loss = romnet.GAP_loss(X_pred, X, G)
        # reg = gamma * autoencoder.regularizer()
        return loss  # + reg

    # train autoencoder
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-----------------")
        romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
        romnet.test_loop(test_dataloader, autoencoder, loss_fn)

    # save autoencoder
    romnet.save_romnet(autoencoder, basename + train_num + ".romnet")

    print("Done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train_autoencoder("noack")
    elif len(sys.argv) == 2:
        train_autoencoder(sys.argv[1])
