import numpy as np
from mpi4py import MPI

# from edpyt.dedlib import DED

lorentz = lambda z: 1/np.pi * (0.15)/(z**2+0.15**2) #Gamma==0.3

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    energies = np.linspace(-1,1,11,endpoint=True)
    eta = 0.02
    solver = DED(lorentz, energies, eta, lorenz_broad=True)

    np.random.seed(0)
    solver.run(N=10)

    sendbuf = solver.sigma
    if rank == 0:
        recvbuf = np.empty_like(sendbuf)
    else:
        recvbuf = None

    comm.Reduce([sendbuf, MPI.COMPLEX], [recvbuf, MPI.COMPLEX],
                op=MPI.SUM, root=0)

    if rank == 0:
        np.testing.assert_allclose(solver.sigma, recvbuf/size)

if __name__ == '__main__':
    main()
