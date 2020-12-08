#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, "/home/ggandus/Libraries/edpyt")

import numpy as np
import pickle
from mpi4py import MPI

from matplotlib import pyplot as plt

from edpyt.espace import build_espace
from edpyt.gf_lanczos import build_gf_lanczos
from edpyt.dedlib import ded_solve, smooth

import random


def lorentzian_function(gamma, z0):
    def inner(z):
        return 1/np.pi * (0.5*gamma) / ((z-z0)**2 + (0.5*gamma)**2)
    inner.z0 = z0
    inner.gamma = gamma
    return inner

e0 = 0.
dos = lorentzian_function(2*0.3, e0)
#dos = pickle.load(open('dos_interp.pckl','rb'))
eta = 0.02
energies = np.load('/home/gag/ownCloud/PTM/nospin/DED/mesh_pm5.npy')
wr = energies
wi = eta*np.abs(wr)
w = wr + 1.j*wi


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(10)
U = 3.

tol = 1e-3
max_it = 3
it = 0
sigma0 = U/4.

if rank==0: 
    import sys
    from pathlib import Path
    f=open('lorentz_sigma0.txt','w')
    rootdir=Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd()
    rootdir.mkdir(parents=True,exist_ok=True)

while (it<max_it):# and (abs(eps)>tol):

    if it>0:
        sigma0 = comm.bcast(sigma0, root=0)

    ss = np.random.SeedSequence(entropy=235693412236239200271790666654757833939, spawn_key=(rank,))
    random.seed(ss.generate_state(1)[0])

    sigma = ded_solve(dos, w, sigma0=sigma0, U=U, n=8, N=N, beta=1e4, rng=random) #rng)

    sendbuf = sigma/N
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty_like(sendbuf)

    comm.Reduce([sendbuf, MPI.COMPLEX], [recvbuf, MPI.COMPLEX],
                op=MPI.SUM, root=0)

    if rank==0:
        sigma = recvbuf/size
        sigma0 = smooth(wr, sigma)(0.).real + sigma0
        f.write(f'{sigma0}\n'); f.flush()
        np.save(rootdir/f'lorentz_sigma_{it}', sigma)
        gf = 1 / (w -e0 -sigma + 0.3j)
        plt.plot(wr, -1/np.pi * gf.imag)
        plt.savefig(rootdir/f'lorentz_ded_{it}.png', dpi=300)
        plt.close()

    it += 1
if rank==0: f.close()
# In[ ]:
