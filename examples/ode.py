import numpy as np
import matplotlib.pyplot as plt
import romnet


class Oscillator(romnet.Model):
    state_dim = 2

    def rhs(self, x):
        return np.array([x[1], -x[0]])

    def adjoint_rhs(self, x, v):
        return np.array([-v[1], v[0]])


class Oscillator2(romnet.SemiLinearModel):
    state_dim = 2
    _linear = np.array([[0, -1], [1, 0]])

    def linear(self, x):
        return self._linear @ x

    def nonlinear(self, x):
        return np.array([0, 0])

    def get_solver(self, alpha):
        mat = np.eye(2) - alpha * self._linear
        return romnet.LUSolver(mat)


def bilinear_example(beta):
    c = np.zeros(3)
    L = np.diag([-1, -2, -5])
    B = np.zeros((3, 3, 3))
    B[0, 0, 2] = beta  # x z term in the first equation
    B[1, 1, 2] = beta  # y z term in the second equation
    return romnet.BilinearModel(c, L, B)


def simulate(scheme, model, x0, Tmax, xf):
    npts = 2**np.arange(5, 11)
    plt.figure()
    prev_err = 0
    for n in npts:
        x = np.zeros((n+1, model.state_dim))
        dt = Tmax / n
        step = model.get_stepper(dt, method=scheme)
        x[0] = x0
        for i in range(n):
            x[i+1] = step(x[i])
        plt.plot(dt * np.arange(n+1), x[:, 0], label=f"n = {n}")
        err = x[n] - xf
        err = np.sqrt(np.dot(err, err))
        print("  %6d  %12g  %2.5f" % (n, err, prev_err / err))
        prev_err = err
    plt.xlabel("Time")
    plt.title(f"Scheme: {scheme}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    oscillator = [Oscillator2(), [1, 0], 2 * np.pi, [1, 0]]
    # final state for bilinear model, at t = 1
    xf = [19.5514257469239, 7.19256757788333, 0.00673794699908548]
    bilinear = [bilinear_example(20), [1, 1, 1], 1.0, xf]
    for case in [oscillator, bilinear]:
        for scheme in romnet.Timestepper.methods():
            print(scheme)
            simulate(scheme, *case)
        for scheme in romnet.SemiImplicit.methods():
            print(scheme)
            simulate(scheme, *case)
