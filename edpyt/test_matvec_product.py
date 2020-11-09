import numpy as np
from scipy.sparse import csr_matrix

from matvec_product import (
    matvec_operator
)


def test_matvec_product():

    dup = 50
    dwn = 70

    Hup = np.random.random((dup,dup))
    Hdw = np.random.random((dwn,dwn))
    Hdd = np.random.random(dup*dwn)
    Iup = np.eye(dup)
    Idw = np.eye(dwn)
    H = np.diag(Hdd) + np.kron(Idw,Hup) + np.kron(Hdw,Iup)

    vec = np.random.random(dup*dwn)

    matvec = matvec_operator(Hdd, Hup, Hdw)
    assert np.allclose(H.dot(vec), matvec(vec))

    sp_Hup = csr_matrix(Hup)
    sp_Hdw = csr_matrix(Hdw)
    sp_matvec = matvec_operator(Hdd, Hup, Hdw)
    assert np.allclose(H.dot(vec), sp_matvec(vec))
