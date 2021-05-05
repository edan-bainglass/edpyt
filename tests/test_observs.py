import pytest
import numpy as np
np.random.seed(0)
from edpyt.observs import get_evecs_occupation

def build_states(n):
    sup = np.empty(n,np.uint32)
    for k in range(n):
        sup[k] = np.uint32(1)<<k
    sdw = sup
    return sup, sdw


def test_get_evecs_occupation():
    n = np.random.choice(np.arange(1,20), size=1)[0]
    sup, sdw = build_states(n)
    for order in ['C','F']:
        eigvecs = np.ones((n*n,3),order=order)/np.sqrt(n)
        nup, ndw = get_evecs_occupation(eigvecs, sup, sdw, n)
        np.testing.assert_allclose(nup.mean(0),1)
        np.testing.assert_allclose(ndw.mean(0),1)


@pytest.mark.skip("MPI skip")
def test_get_evec_occpupation_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    n_local = 5
    n = n_local*comm.size
    sup, sdw = build_states(n)
    for order in ['C','F']:
        eigvecs = np.ones((n*n_local,3),order=order)/np.sqrt(n)
        nup, ndw = get_evecs_occupation(eigvecs, sup, sdw, n, comm=comm)
        np.testing.assert_allclose(nup.mean(0),1)
        np.testing.assert_allclose(ndw.mean(0),1)


if __name__=='__main__':
    test_get_evec_occpupation_mpi()