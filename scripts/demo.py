import numpy as np
import torch
from torch.utils.data import DataLoader
import romnet
from noack import NoackModel, random_ic

def main():
    model = NoackModel(mu=0.1, omega=2., A=-0.1, lam=10)
    dt = 0.1
    model.set_stepper(dt, method="rk2", nsteps=5)

    # generate trajectories for training/testing
    num_train = 1024
    num_test = 64
    n = 30 # length of each trajectory
    print("Generating training trajectories...")
    training_traj = romnet.sample(model.step, random_ic, num_train, n)
    test_traj = romnet.sample(model.step, random_ic, num_test, n)

    # sample gradients for GAP loss
    s = 32 # samples per trajectory
    L = 15 # horizon for gradient sampling
    print("Sampling gradients...")
    training_data = romnet.sample_gradient(training_traj, model, s, L)
    test_data = romnet.sample_gradient(test_traj, model, s, L)
    print("Done")

    # train the autoencoder
    learning_rate = 1.e-3
    batch_size = 64
    num_epochs = 50
    dims = [3, 3, 3, 3, 3, 2]
    autoencoder = romnet.ProjAE(dims)
    loss_fn = romnet.GAP_loss
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-----------------")
        romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
        romnet.test_loop(test_dataloader, autoencoder, loss_fn)

if __name__ == "__main__":
    main()
