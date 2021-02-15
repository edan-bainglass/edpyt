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
import logging
import time
import functools

def lorentzian_function(gamma, z0):
    def inner(z):
        return 1/np.pi * (0.5*gamma) / ((z-z0)**2 + (0.5*gamma)**2)
    inner.z0 = z0
    inner.gamma = gamma
    return inner

#dos = pickle.load(open('dos_interp.pckl','rb'))
eta = 0.02
energies = np.load('/home/gag/Projects/lorentz_ded/data/mesh_pm5.npy')
wr = energies
wi = eta*np.abs(wr)
w = wr + 1.j*wi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 5 #int(5e3)
U = 3.

tol = 1e-3
max_it = 2
sigma0 = U/2.
ed = -2
beta = 0.1

import sys
from pathlib import Path
rootdir=Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd()

if rank==0:
    rootdir.mkdir(parents=True,exist_ok=True)
    f=open(rootdir/'lorentz_sigma0.txt','w')

# Split random sequence of poles
my_N = N//size 
if rank==(size-1): my_N += N%size
def init_rng(rng, seed, N):
    random.seed(seed)
    for _ in range(N):
        rng.random()
#consider 50% success, consume x2 sequence lenght xn poles
n = 4
my_start = 2*n*(N//size)*rank 

ss = np.random.SeedSequence(entropy=235693412236239200271790666654757833939)
seed = ss.generate_state(1)[0]

logging.basicConfig(filename=rootdir/'ded.log', 
                    filemode='a',
                    format=f"[{rank}] %(levelname)s %(message)s",
                    level=logging.INFO)
logging.info("Start DED loop")

def timer(f):
    functools.wraps(f)
    def _inner(*args, **kwargs):
        start_time = time.perf_counter()
        logging.info(f"Start {f.__name__}")
        results = f(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        logging.info(f"Finished {f.__name__} in {run_time} secs")
        return results
    return _inner

ded_solve = timer(ded_solve)
init_rng = timer(init_rng)

it = 0
eps = tol + 1.
while (it<max_it) and (abs(eps)>tol):

    dos = lorentzian_function(2*0.3, ed+sigma0)
    sigma = np.zeros(w.size+3, dtype=w.dtype)
    
    init_rng(random, seed, my_start)
    
    imp_occp0, imp_occp1, imp_entropy = ded_solve(
        dos, w, sigma=sigma[:-3], sigma0=sigma0, U=U, n=n,
        N=my_N, beta=beta, rng=random)
    
    sigma[-3] = imp_occp0
    sigma[-2] = imp_occp1
    sigma[-1] = imp_entropy
    sendbuf = sigma
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty_like(sendbuf)
    
    comm.Reduce([sendbuf, MPI.COMPLEX16], [recvbuf, MPI.COMPLEX16],
                op=MPI.SUM, root=0)
    
    if rank==0:
        sigma = recvbuf[:-3]/size
        imp_occp0 = recvbuf[-3].real/size
        imp_occp1 = recvbuf[-2].real/size
        imp_entropy = recvbuf[-1].real/size
        eps = smooth(wr, sigma)(0.).real
        f.write(f'{sigma0:.5f} {eps:.5f} {imp_occp0:.5f} {imp_occp1:.5f} {imp_entropy:.5f}\n'); f.flush()
        np.save(rootdir/f'lorentz_sigma_{it}', sigma)
        gf = 1 / (w -ed -sigma -sigma0 + 0.3j)
        plt.plot(wr, -1/np.pi * gf.imag)
        plt.savefig(rootdir/f'lorentz_ded_{it}.png', dpi=300)
        plt.close()
    
    sigma0 += comm.bcast(eps, root=0)
    it += 1

if rank==0: 
    f.close()
# In[ ]:
