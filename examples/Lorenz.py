import numpy as np
import scipy as sp

import romnet

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class LorenzSystem(romnet.BilinearModel):
    def __init__(self, n: int, T: float, sigma: float, rho: float, beta: float):
        """summary here
        Args:
            n: Number grid points
            L: Length of time
            sigma, rho, and beta: parameters in the Lorenz system
        """
        self.n = n
        self.T = T
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        
        self._affine = np.zeros(3)
        
        # matrix that contains the linear relation
        L = np.diag([-sigma, -1, -beta])
        L[0,1] = sigma
        L[1,0] = rho
        self._linear = L
        
        # rank-3 tensor describing the bilinear map B
        B = np.zeros((3, 3, 3))
        B[1, 0, 2] = -1  # -xz term in the second equation
        B[2, 0, 1] = 1  # xy term in the third equation
        self._bilinear = B
        

def main():
    import matplotlib.pyplot as plt

    n = 1000
    T = 100
    rho = 28
    sigma = 10
    beta = 8/3
    lorenz = LorenzSystem(n,T,sigma,rho,beta)
    dt = T/n
    lorenz.set_stepper(dt, "rk4")

    # initial condition
    x_vec = [1,1,1]

    sol = np.zeros((n + 1, 3))
    sol[0,:] = x_vec # this is the initial condition

    for t in range(1,n+1):
        x_vec = lorenz.step(x_vec)
        sol[t,:] = x_vec

    
    ax = plt.axes(projection='3d')

    ax.scatter3D(sol[:,0], sol[:,1], sol[:,2], c = sol[:,2], cmap='Greens');
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz system simulation, T = {T}")
    plt.show()

if __name__ == "__main__":
    main()
