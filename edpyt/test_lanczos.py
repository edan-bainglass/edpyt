import numpy as np

from lanczos import (
    build_sl_tridiag,
    egs_tridiag
)

def test_build_sl_tridiag():
    n = 30
    d = np.random.random(n)
    e = np.random.random(n-1)
    mat = np.diag(d) + np.diag(e,1) + np.diag(e,-1)
    mat += mat.T
    mat /= 2

    expected = np.linalg.eigvalsh(mat)[0]

    a, b = build_sl_tridiag(mat.dot, np.random.random(n))
    assert np.allclose(
        expected,
        egs_tridiag(a, b[1:])
    )
