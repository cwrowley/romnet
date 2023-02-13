import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
        
def simulate(n,T,x0,lorenz):
    x_vec = x0  # the initial condition
    
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
    
    return sol
    
def simpson(h,f):
    """Simpson's 1/3 rule
    Args:
        h: (b-a)/2
        f is the array that contains the value of f at
        a, (a+b)/2 and b.
    """
    f_a = f[0]
    f_ab = f[1]
    f_b = f[2]
    return (1/3)*h*(f_a + 4*f_ab + f_b)
    
def Dg():
    """This function returns the Jacobian matrix of g,
    taking z in the Lorenz system to be the output.
    """
    return np.array([0,0,1])

def Df(xyz,dt,sigma,rho,beta):
    """Returns the Jacobian matrix of f at time t
    Args:
        xyz: 3 by 3 matrix, containing the value of x, y, and z at time t, 
    t + delta_t/2 and t + delta_t. Has the form [x_vec(t);x_vec(t + delta_t/2);x_vec(t + delta_t)].
        dt: delta t
        sigma, rho, and beta: parameters in the Lorenz system
    """
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    
    D = np.diag([-sigma*dt, -dt, -beta*dt])
    D[0,1] = sigma*dt
    D[1,0] = rho*dt-simpson(dt/2,z)
    D[1,2] = -simpson(dt/2,x)
    D[2,0] = simpson(dt/2,y)
    D[2,1] = simpson(dt/2,x)
    return D

def assemble_gradientcov(sol,dt,sigma,rho,beta):
    """Returns gradient covariance matrix"""
    n = np.shape(sol)[0]-1
    D = np.zeros((int(n/2),3))
    
    i = 0
    while i < n:
        DF = Dg() 
        j = i
        while j >= 0:
            DF = DF@Df(sol[j:j+3,:],dt,sigma,rho,beta)
            j = j - 2
        D[int(i/2),:] = DF
        i = i + 2
        
    return D.T@D

    
def main():

    n = 1000
    T = 50
    rho = 28
    sigma = 10
    beta = 8/3
    lorenz = LorenzSystem(n,T,sigma,rho,beta)
    dt = T/n
    lorenz.set_stepper(dt, "rk4")

    # initial condition
    x_vec = [1,1,1]
    
    sol = simulate(n,T,x_vec,lorenz)
    grad = assemble_gradientcov(sol,dt,sigma,rho,beta)
    print(grad)

if __name__ == "__main__":
    main()
