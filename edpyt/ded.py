#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, "/home/ggandus/Libraries/edpyt")

import numpy as np
from mpi4py import MPI

from matplotlib import pyplot as plt

from edpyt.espace import build_espace
from edpyt.gf_lanczos import build_gf_lanczos
from dedlib import *

# In[2]:


def lorentzian_function(gamma, z0):
    def inner(z):
        return 1/np.pi * (0.5*gamma) / ((z-z0)**2 + (0.5*gamma)**2)
    inner.z0 = z0
    inner.gamma = gamma
    return inner


lorentz = lorentzian_function(2*0.3, 0.)
eta = 0.02
energies = np.load('/home/ggandus/Libraries/edpyt/ipynbs/PTM/mesh_pm5.npy')
wr = energies
wi = eta*np.abs(wr)
w = wr + 1.j*wi
#sigma = np.zeros_like(w)

from edpyt.shared import params
params['hfmode'] = False #True



def solve(dos, z, sigma=None, sigma0=None, n=4, N=int(1e3), U=3., rng=np.random):
    _sigma = sigma or np.zeros_like(z)
    _sigma0 = sigma0 or U/2.
    H = np.zeros((n,n))
    V = np.zeros((n,n))
    neig = np.ones((n+1)*(n+1)) * 1
    for _ in range(N):
        found = False
        while not found:
            poles = sample(dos, 10., 0., n, rng=rng)
            gf0 = build_gf0(poles)
            gfimp = build_gfimp(gf0)
            build_siam(H, V, 0., gfimp)
            espace, egs = build_espace(H, V, neig)
            keep_gs(espace, egs)
            sct = next(v for v in espace.values())
            if sct.eigvecs.ndim < 2: continue
            evec = sct.eigvecs[:,0]
            N0 = get_occupation(evec,sct.states.up,sct.states.dw,n)
            V[0,0] = U
            H[0,0] -= _sigma0
            espace, egs = build_espace(H, V, neig)
            keep_gs(espace, egs)
            sct = next(v for v in espace.values())
            evec = sct.eigvecs[:,0]
            Nv = get_occupation(evec,sct.states.up,sct.states.dw,n)
            if np.allclose(Nv,N0):
                gf = build_gf_lanczos(H, V, espace, 0.)
                _sigma += np.reciprocal(gf0(z))-np.reciprocal(gf(z.real,z.imag))
                found = True
    if sigma is None:
        return _sigma


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ss = np.random.SeedSequence(entropy=235693412236239200271790666654757833939, spawn_key=(rank,))
rng = np.random.default_rng(ss)

N = int(1e4)
sigma = solve(lorentz, w, N=N, rng=rng)

sendbuf = sigma/N
recvbuf = None
if rank == 0:
    recvbuf = np.empty_like(sendbuf)

comm.Reduce([sendbuf, MPI.COMPLEX], [recvbuf, MPI.COMPLEX],
            op=MPI.SUM, root=0)

if rank==0:
    sigma = recvbuf/size
    gf = 1 / (w - sigma + 0.3j)
    plt.plot(wr, -1/np.pi * gf.imag)
    plt.savefig('lorentz_ded.png', dpi=300)
    plt.close()

# In[ ]:




