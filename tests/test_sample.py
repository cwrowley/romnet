import numpy as np
from romnet.sample import sample, sample_gradient, TrajectoryList
from romnet import Model
import torch

__all__ = ["sample", "sample_gradient"]

class MyModel(Model):
    def __init__(self, dim=1, debug=False):
        self.debug = debug
        self.state_dim = dim
        self.output_dim = dim

    def rhs(self, x):
        if self.debug:
            print(f"  rhs({x})")
        return -x**3

    def adjoint_rhs(self, x, v):
        if self.debug:
            print(f"  adjoint_rhs({x}, {v})")
        return -3 * x**2 * v

def try_steppers():
    dim = 2
    model = MyModel(dim=dim, debug=False)
    dt = 0.1
    model.set_stepper(dt, method="rk2", nsteps=1)
    x = np.ones(dim)
    v = np.ones(dim)
    for i in range(5):
        x = model.step(x)
        print(x)
        v = model.adjoint_step(x,v)

def test_trajlist():
    a = np.arange(60).reshape(3,5,2,2)
    b = TrajectoryList(a)
    assert(b[0].shape == (2,2))
    assert(b.traj[0].shape == (5,2,2))

def try_sample():
    dim = 2
    model = MyModel(dim=dim, debug=False)
    dt = 0.1
    model.set_stepper(dt, method='rk2')
    # stepper = timestepper(model, dt)
    num_traj = 50
    n = 25
    random_ic = lambda : np.random.randn(dim)
    data = sample(model.step, random_ic, num_traj, n)
    # for i,x in enumerate(data.traj):
        # print("Trajectory %d\n-------------" % i)
        # print(x)
    samples_per_traj = 10

    s = 10 # samples per trajectory
    L = 20 # horizon for gradient sampling
    grad_data = sample_gradient(data, model, s, L)

    dataloader = torch.utils.data.DataLoader(grad_data, batch_size=5, shuffle=True)
    print(dataloader)
    X, G = next(iter(dataloader))
    print(X.size())
    print(X)
    print(G.size())
    print(G)

if __name__ == "__main__":
    # try_steppers()
    # test_trajlist()
    try_sample()
    # try_sample_grad()
