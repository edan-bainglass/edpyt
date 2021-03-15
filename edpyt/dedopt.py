
# from edpyt.ded import init_rng
import numpy as np
from scipy import optimize as opt

from mpi4py import MPI
from edpyt.parallel import *

x0 = 0.3 # !! Must be the same for all ranks.
x = np.ones(10,complex)

sendbuf = x
recvbuf = np.empty_like(x)

def f(x0, c):
    comm.Reduce([sendbuf, MPI.COMPLEX16], [recvbuf, MPI.COMPLEX16],
                 op=MPI.SUM)
    
    eps = None
    if comm.rank == 0:
        eps = (recvbuf - x0).sum().real
    
    eps = comm.bcast(eps, root=0)
    
    return eps

# file = paropen('tmp.txt')

# def callback(x, f):
#     parprint(x, file=file)

res = opt.root(f, 0., 1., options=dict(xtol=1e-3,maxfev=10))

print(comm.rank, res.x[0], res)

# class Optimizer:

#     def __init__(self, opt, f, callback) -> None:
#         self.opt = opt
#         self.f = f
#         self.callback = callback

#     def callback(self, x):
#         """Callback function to be run after each iteration by SciPy

#         This should also be called once before optimization starts, as SciPy
#         optimizers only calls it after each iteration.
#         """
#         self._callback()

#     def     

#     def run(self, x0):
#         self.callback()

    