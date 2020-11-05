import numpy as np

from build_mb_ham import (
    build_mb_ham
)


def test_build_mb_ham():

    t = -1

    H = np.array([
        [0,t],
        [t,0]
    ])

    V = np.array([
        [0,0],
        [0,0]
    ])

    i_vals, iup_mat, idw_mat = build_mb_ham(H, V, 1, 1)

    dup = iup_mat.shape[0]
    dwn = idw_mat.shape[0]

    mb_ham = np.diag(i_vals) \
             + np.kron(iup_mat.todense(), np.eye(dwn)) \
             + np.kron(np.eye(dup), idw_mat.todense())

    np.savetxt('mb_ham.txt', mb_ham)
