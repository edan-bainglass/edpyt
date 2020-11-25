import numpy as np


from lanczos import (
    build_sl_tridiag,
    sl_step
)

from tridiag import (
    egs_tridiag,
    eigh_tridiagonal,
)


n = 30
d = np.random.random(n)
e = np.random.random(n-1)
mat = np.diag(d) + np.diag(e,1) + np.diag(e,-1)
mat += mat.T
mat /= 2


def test_build_sl_tridiag_egs():
    global n
    v0 = np.random.random(n)
    a, b = build_sl_tridiag(mat.dot, v0)

    # TEST egs
    expected = np.linalg.eigvalsh(mat)[0]
    computed = egs_tridiag(a, b[1:])
    assert np.allclose(
        expected,
        computed
    )

    # TEST gs
    expected = np.linalg.eigh(mat)[1]

    _, U = eigh_tridiagonal(a, b[1:])
    computed = np.zeros((v0.size,a.size))

    v = v0
    l = np.zeros_like(v)
    for n in range(a.size):
        _, _, l, v = sl_step(mat.dot, v, l)
        computed += l[:,None] * U[n][None,:]

    assert np.allclose(
        np.abs(expected[:,0]),
        np.abs(computed[:,0])
    )
