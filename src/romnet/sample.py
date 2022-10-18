import numpy as np

__all__ = ["sample", "sample_gradient"]

class TrajectoryList:
    """
    Container for samples of trajectories

    Suppose traj is a numpy array of dimension (num_traj, n, *)
    That is, there are num_traj trajectories, each of length n

    dataset = TrajectoryList(traj)
    dataset.traj[i] is trajectory i (an array with n samples)
    dataset[j] is sample j (from all trajectories, concatenated together)

    This class is compatible with torch.DataLoader:

    training = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, traj):
        self.traj = np.array(traj)
        self.num_traj = self.traj.shape[0]
        self.n = self.traj.shape[1]
        newshape = list(self.traj.shape)
        newshape[1] *= newshape[0]
        newshape.pop(0)
        self.data = self.traj.view()
        self.data.shape = newshape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class GradientDataset:
    def __init__(self, X, G):
        self.X = np.array(X)
        self.G = np.array(G)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.G[i]


def sample(step, random_state, num_traj, n):
    """
    Sample num_traj trajectories each with length n

    random_state() generates a random initial state x
    step(x) advances the state forward in time

    Returns a TrajectoryList object
    """
    traj_list = list()
    for i in range(num_traj):
        traj = list()
        x = random_state()
        traj.append(x)
        for t in range(n-1):
            x = step(x)
            traj.append(x)
        traj_list.append(traj)
    return TrajectoryList(traj_list)

def sample_gradient(traj_list, model, samples_per_traj, L):
    """
    Cobras-style gradient sampling

    traj_list is a lost of trajectories (e.g., TrajectoryList object)
    adj_step(x, v) advances the adjoint variable v at state x
    samples_per_traj is the number of gradient samples for each trajectory
    L is the horizon used for advancing the adjoint variable
    """
    X = list()
    G = list()
    N = traj_list.n  # num pts in each trajectory
    for k, x in enumerate(traj_list.traj):
        for j in range(samples_per_traj):
            # choose a time t in [0..N-1-L]
            t = np.random.randint(N - L)
            # choose a tau in [0..L]
            tau = np.random.randint(L + 1)
            # choose random direction eta for gradient
            eta = np.sqrt(L + 1) * np.random.randn(model.output_dim)
            lam = model.adjoint_output(x[t + tau], eta)
            for i in range(1, tau):
                lam = model.adjoint_step(x[t + tau - i], lam)
            X.append(x[t])
            G.append(lam)
    return GradientDataset(X, G)
