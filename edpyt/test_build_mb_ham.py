import numpy as np

from build_mb_ham import (
    build_mb_ham
)


t12 = -1.
t21 = -1.
U1 = 1.
U2 = 0.
e1 = -0.5
e2 = -0.25

H = np.array([
    [e1,t12],
    [t21,e2]
])

V = np.array([
    [U1,0],
    [0,U2]
])

def test_build_mb_ham():

    # sz = 0 sector

    nup = 1
    ndw = 1

    i_vals, iup_mat, idw_mat = build_mb_ham(H, V, nup, ndw)

    dup = iup_mat.shape[0]
    dwn = idw_mat.shape[0]

    mb_ham = np.diag(i_vals) \
             + np.kron(iup_mat.todense(), np.eye(dwn)) \
             + np.kron(np.eye(dup), idw_mat.todense())

    # states :: 01, 10

    #
    # state :   |up,dw>   |up,dw>
    # ----------------------------
    #  s0   :     0,0      1,1
    #  s1   :     1,0      0,1
    #  s2   :     0,1      1,0
    #  s3   :     1,1      0,0

    # Hamiltonian

    #           s0       s1      s2      s3
    #
    #   s0   2*e2+U2 | -t21  |  -t21  |   0
    #        ---------------------------------
    #   s1    -t12   | e1+e2 |    0   |  -t21
    #        ---------------------------------
    #   s2    -t12   |    0  |  e1+e2 |  -t21
    #        ---------------------------------
    #   s3      0    |  -t12 |   -t12  | 2*e1+U1

    expected = np.array([
        [2*e2+U2,-t21,-t21,0],
        [-t12,e1+e2,0,-t21],
        [-t12,0,e1+e2,-t21],
        [0,-t12,-t12,2*e1+U1]
    ])

    assert np.allclose(mb_ham, expected)


    # sz = 1 sector

    nup = 2
    ndw = 0

    i_vals, iup_mat, idw_mat = build_mb_ham(H, V, nup, ndw)

    dup = iup_mat.shape[0]
    dwn = idw_mat.shape[0]

    mb_ham = np.diag(i_vals) \
             + np.kron(iup_mat.todense(), np.eye(dwn)) \
             + np.kron(np.eye(dup), idw_mat.todense())

    #
    # state :   |up,dw>   |up,dw>
    # ----------------------------
    #  s0   :     1,0      1,0

    # Hamiltonian

    #           s0
    #
    #   s0   e1+e2

    expected = np.array([
        [e1+e2]
    ])

    assert np.allclose(mb_ham, expected)

    # sz = -1 sector

    nup = 0
    ndw = 2

    i_vals, iup_mat, idw_mat = build_mb_ham(H, V, nup, ndw)

    dup = iup_mat.shape[0]
    dwn = idw_mat.shape[0]

    mb_ham = np.diag(i_vals) \
             + np.kron(iup_mat.todense(), np.eye(dwn)) \
             + np.kron(np.eye(dup), idw_mat.todense())

    #
    # state :   |up,dw>   |up,dw>
    # ----------------------------
    #  s0   :     0,1      0,1

    # Hamiltonian

    #           s0
    #
    #   s0   e1+e2

    expected = np.array([
        [e1+e2]
    ])

    assert np.allclose(mb_ham, expected)
