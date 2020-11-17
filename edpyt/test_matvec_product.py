import numpy as np
from scipy.sparse import csr_matrix
import time

from matvec_product import (
    matvec_operator
)


def test_matvec_product():

    dup = 100
    dwn = 200

    Hup = np.random.random((dup,dup))
    Hdw = np.random.random((dwn,dwn))
    Hdd = np.random.random(dup*dwn)
    Iup = np.eye(dup)
    Idw = np.eye(dwn)
    H = np.diag(Hdd) + np.kron(Idw,Hup) + np.kron(Hdw,Iup)

    vec = np.random.random(dup*dwn)

    # matvec = matvec_operator(Hdd, Hup, Hdw)
    # assert np.allclose(H.dot(vec), matvec(vec))

    sp_Hup = csr_matrix(Hup)
    sp_Hdw = csr_matrix(Hdw)
    sp_matvec = matvec_operator(Hdd, sp_Hup, sp_Hdw)
    s = time.time()
    res = sp_matvec(vec)
    e = time.time()
    print(e-s)
    assert np.allclose(H.dot(vec), res)
