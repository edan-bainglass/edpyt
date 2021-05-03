import pytest
pytest.mark.skip("skipping MPI transpose hilbert.",allow_module_level=True)

import numpy as np
from scipy.sparse import random
# Both numpy and scipy will start from the same seed.
# Hence, all processors will have same random matrices/vectors.
np.random.seed(0)

from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

dup = 4
dwn = 4
dwn_local = dwn//size
dup_local = dup//size

Hup = random(dup,dup,density=0.3,format='csr')
Hdw = random(dwn,dwn,density=0.3,format='csr')
Hdd_local = np.random.random(dup*dwn_local)
vec_local = np.random.random(dup*dwn_local)

tile = lambda a_local: np.tile(a_local, size)
def Gather(a_local):
    a = None
    if rank==0:
        a = np.empty(dup*dwn)
    comm.Gather([a_local,dup*dwn_local,MPI.DOUBLE],
                [a,dup*dwn_local,MPI.DOUBLE],root=0)
    return a

vec = tile(vec_local)
Hdd = tile(Hdd_local)

def test_diag():
    res_local = vec_local * Hdd_local
    res = Gather(res_local)
    if rank == 0:
        expected = vec * Hdd
        np.testing.assert_allclose(res, expected)

def test_up():
    from edpyt._psparse import UPmultiply
    res_local = np.zeros_like(vec_local)
    UPmultiply(Hup, vec_local, res_local)
    res = Gather(res_local)
    if rank == 0:
        expected = Hup.dot(vec.reshape(dwn,dup).T).T.reshape(-1,)
        np.testing.assert_allclose(res, expected)

def test_dw():
    from edpyt._psparse import DWmultiply
    from edpyt.vector_transpose import collect_dw, collect_up
    res_local = np.zeros_like(vec_local)
    r = res_local.reshape(dwn_local,dup)
    v = vec_local.reshape(dwn_local,dup)
    r = collect_dw(r,comm)
    v = collect_dw(v,comm)
    DWmultiply(Hdw, v.reshape(-1), r.reshape(-1,))
    r = collect_up(r,comm).reshape(-1,)
    v = collect_up(v,comm).reshape(-1,)
    res = Gather(r)
    if rank == 0:
        expected = Hdw.dot(vec.reshape(dwn,dup)).reshape(-1,)
        np.testing.assert_allclose(res, expected)


def test_matvec_product():
    from edpyt.matvec_product import matvec_operator
    sp_matvec_mpi = matvec_operator(Hdd_local, Hup, Hdw, comm)
    res_local = sp_matvec_mpi(vec_local)
    res = Gather(res_local)
    if rank==0:
        sp_matvec = matvec_operator(Hdd, Hup, Hdw)
        expected = sp_matvec(vec)
        np.testing.assert_allclose(expected,res)


test_diag()
test_up()
test_dw()
test_matvec_product()