import numpy as np
import pickle

__all__ = ["sample", "sample_gradient", "load"]


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

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class GradientDataset:
    def __init__(self, X, G):
        self.X = np.array(X)
        self.G = np.array(G)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.G[i]

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


def load(fname):
    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    return data


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


def sample_gradient(traj_list, model, samples_per_traj, L,
                    long_traj=False, CoBRAS_style=False):
    """
    Gradient sampling

    Running this method with default settings uses the adjoint sampling method
    seen in Section 3 of the CoBRAS paper (long_traj = False). Additionally,
    this method will output a GradientDataset, which is compatible with
    PyTorch's dataloader.

    If long_traj = True, then Algorithm 3.1 in the CoBRAS paper is used.

    If CoBRAS_style = True, then this method returns a tuple. The first element
    is the GradientDataset. The second element is the gradient matrix Y. Note,
    Y depends the truth value of long_traj. See Equation 3.7 and Algorithm 3.1
    in the CoBRAS paper.

    traj_list is a list of trajectories (e.g., TrajectoryList object)
    adj_step(x, v) advances the adjoint variable v at state x
    samples_per_traj is the number of gradient samples for each trajectory
    L is the horizon used for advancing the adjoint variable
    """
    X = list()
    G = list()
    N = traj_list.n  # num pts in each trajectory
    if long_traj is False:
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
        if CoBRAS_style is False:
            return GradientDataset(X, G)
        else:
            Y = np.array(G) / np.sqrt(len(G))
            return GradientDataset(X, G), Y
    else:
        Y = list()
        for k, x in enumerate(traj_list.traj):
            for j in range(samples_per_traj):
                t = np.random.randint(N - L)
                tau = np.random.randint(L + 1)
                eta = np.sqrt(L + 1) * np.random.randn(model.output_dim)
                tau_min = np.max((0, t + tau - (N - L - 1)))
                tau_max = np.min((L, t + tau))
                nu = 1 + tau_max - tau_min
                X_rev = list()
                Lam = list()
                X_rev.append(x[t + tau])
                Lam.append(model.adjoint_output(x[t + tau], eta))
                for i in range(1, tau_max):
                    X_rev.append(x[t + tau - i])
                    Lam.append(model.adjoint_step(x[t + tau - i], Lam[i - 1]))
                X.extend(X_rev[tau_min:tau_max])
                G.extend(Lam[tau_min:tau_max])
                Y.extend(Lam[tau_min:tau_max] / np.sqrt(nu))
        if CoBRAS_style is False:
            return GradientDataset(X, G)
        else:
            return GradientDataset(X, G), np.array(Y)
