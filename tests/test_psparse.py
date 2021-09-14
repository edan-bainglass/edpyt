import numpy as np
import scipy

from edpyt import _psparse


def test_psparse_UPmultiply():
    n = 3
    m = 2

    A = scipy.sparse.random(m, m, density=0.4, format='csr')
    w = np.random.random(m*n)
    W = w.reshape(n,m)
    result = np.zeros_like(w)

    _psparse.UPmultiply(A,w,result)
    np.testing.assert_allclose(result, A.dot(W.T).T.flatten())

def test_psparse_DWmultiply():
    n = 3
    m = 2

    A = scipy.sparse.random(n, n, density=0.4, format='csr')
    w = np.random.random(m*n)
    W = w.reshape(n,m)
    result = np.zeros_like(w)

    _psparse.DWmultiply(A,w,result)
    np.testing.assert_allclose(result, A.dot(W).flatten())


def test_psparse_Multiply():
    n = 5

    A = scipy.sparse.random(n, n, density=0.4, format='csr')
    w = np.random.random(n)
    result = np.zeros_like(w)

    _psparse.Multiply(A,w,result)
    np.testing.assert_allclose(result, A.dot(w))