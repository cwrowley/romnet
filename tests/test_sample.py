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
        v = model.adjoint_step(x, v)


def test_trajlist():
    a = np.arange(60).reshape(3, 5, 2, 2)
    b = TrajectoryList(a)
    assert b[0].shape == (2, 2)
    assert b.traj[0].shape == (5, 2, 2)


def test_sample():
    dim = 2
    model = MyModel(dim=dim, debug=False)
    dt = 0.1
    model.set_stepper(dt, method='rk2')
    # stepper = timestepper(model, dt)
    num_traj = 5
    n = 10
    def random_ic(): return np.random.randn(dim)
    data = sample(model.step, random_ic, num_traj, n)
    assert len(data) == num_traj * n
    # for i,x in enumerate(data.traj):
    #     print("Trajectory %d\n-------------" % i)
    #     print(x)

    s = 3  # samples per trajectory
    L = 5  # horizon for gradient sampling
    batch_size = 7

    # test gradient sampling default
    grad_data = sample_gradient(data, model, s, L)
    assert len(grad_data) == num_traj * s
    dataloader = torch.utils.data.DataLoader(grad_data, batch_size=batch_size,
                                             shuffle=True)
    X, G = next(iter(dataloader))
    assert X.shape == torch.Size([batch_size, dim])
    assert G.shape == torch.Size([batch_size, dim])

    # test gradient sampling with CoBRAS_style=True
    grad_data, Y = sample_gradient(data, model, s, L, CoBRAS_style=True)
    assert len(grad_data) == num_traj * s
    assert grad_data.X.shape == Y.shape

    # test gradient sampling with long_traj=True
    grad_data = sample_gradient(data, model, s, L, long_traj=True)
    # note, len(grad_data) in this case will be random.

    # test gradient sampling with CoBRAS_style=True and long_traj=True
    grad_data, Y = sample_gradient(data, model, s, L, long_traj=True,
                                   CoBRAS_style=True)
    assert grad_data.X.shape == Y.shape


if __name__ == "__main__":
    # try_steppers()
    # test_trajlist()
    test_sample()
    # try_sample_grad()
