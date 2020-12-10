import numpy as np
import scipy
from scipy.sparse import linalg as sla
from scipy.sparse.linalg.interface import LinearOperator

from edpyt.eigh_arpack import (
    eigsh
)

from edpyt.matvec_product import (
    matvec_operator
)

def test_eigh_arpack():
    m = 20
    n = 10
    Hdd = np.random.random(m*n)
    Hup = scipy.sparse.random(m, m, format='csr', density=0.3)
    Hdw = scipy.sparse.random(n, n, format='csr', density=0.3)

    matvec = matvec_operator(Hdd, Hup, Hdw)

    v0 = np.random.random(m*n)
    w, v = eigsh(m*n, 3, matvec, v0=v0)
    A = LinearOperator((m*n,m*n), matvec, Hdd.dtype)
    w_expected, v_expected = sla.eigsh(A, 3, which='SA', v0=v0)

    np.testing.assert_allclose(w, w_expected)
    np.testing.assert_allclose(v, v_expected)
