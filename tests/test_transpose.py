import pytest
pytest.mark.skip("skipping MPI transpose hilbert.",allow_module_level=True)

import numpy as np
from edpyt.vector_transpose import collect_dw, collect_up
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def test_transpose():
    m = 3
    n = size * 3
    mxn = m*n
    a = (np.arange(mxn)+rank*mxn).reshape(m,n)
    expected = a.copy()
    b = collect_dw(a)
    a = collect_up(b)
    np.testing.assert_allclose(a, expected)

test_transpose()