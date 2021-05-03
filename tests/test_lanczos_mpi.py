import pytest
pytest.mark.skip("skipping MPI transpose hilbert.",allow_module_level=True)

import numpy as np
# All processes have same matrix
np.random.seed(0)
from mpi4py import MPI

from edpyt.lanczos import build_sl_tridiag, sl_solve

comm = MPI.COMM_WORLD

n = 30
d = np.random.random(n)
e = np.random.random(n-1)
H = np.diag(d) + np.diag(e,1) + np.diag(e,-1)
H += H.T
H /= 2

class pmatvec:
    """Simple parallel matrix vector implementation for testing purpose.
    
    The input matrix is partitioned along the 1st dimension, i.e. each 
    process gets all rows and a cunck of consecutive columns.
    """
    def __init__(self, H, comm) -> None:
        n = H.shape[1]
        batch = n//comm.size
        self.H = H[:,comm.rank*batch:(comm.rank+1)*batch]
        self.comm = comm
    
    def __call__(self, v):
        """Matrix vector multiplication.
        
        Multiply the elements of a vector corresponding to
        the cunck of assigned columns. 
        Pseudocode:
            w_i = H_ij v_j, for {j} in my comumns
            all distribute w_j to the correct process
        """
        H = self.H
        comm = self.comm
        
        w = H.dot(v)
        l = None
        batch = v.size
        for i in range(comm.size):
            if i==comm.rank:
                l = np.empty_like(v)
            comm.Reduce([w[i*batch:(i+1)*batch],batch],l,op=MPI.SUM,root=i)
        return l


def test_pmatvec():
    matvec = pmatvec(H, comm)
    batch = n//comm.size
    v = np.ones(batch)
    w = matvec(v)
    res = H.dot(np.ones(n))
    expected = res[comm.rank*batch:(comm.rank+1)*batch]
    np.testing.assert_allclose(expected, w)


def test_build_sl_tridiag_egs():
    global n
    matvec = pmatvec(H, comm)
    batch = n//comm.size
    v0 = np.ones(batch)
    a, b = build_sl_tridiag(matvec, v0, comm=comm)

    # TEST egs and gs
    expected_w, expected_v = np.linalg.eigh(H)
    egs, gs = sl_solve(matvec, a, b, v0, select=2, comm=comm)
    if comm.rank==0:
        print()

    expected_egs = expected_w[0]
    expected_gs = expected_v[comm.rank*batch:(comm.rank+1)*batch,0]
    np.testing.assert_allclose(egs, expected_egs)
    np.testing.assert_allclose(np.abs(gs.flat), np.abs(expected_gs))


test_pmatvec()
test_build_sl_tridiag_egs()
