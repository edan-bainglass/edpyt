#!/usr/bin/env python
# coding: utf-8

import numpy as np

from edpyt.espace import build_espace

"""Extended Hubbard dimer.

"""

U = 0.
J = 0.75

eS = - 3.08
eAS = - 2.8
Delta = eAS - eS


H = np.array([
    [eS,0],
    [0,eAS]
])

V = np.array([
    [U,0],
    [0,U]
])


Jx = np.array([
    [0,J],
    [J,0]
])


Jp = Jx.copy()


# 
eigvals = {
    (0,0):
        (0,),
    (0,1):
        (eS, eS+Delta),
    (1,1): 
        (2*eS + Delta + U - np.sqrt(Delta**2 + J**2),
         2*eS + Delta + U - J,
         2*eS + Delta + U + J,
         2*eS + Delta + U + np.sqrt(Delta**2 + J**2)),
    (1,2):
        (3*eS+Delta+3*U-J,
         3*eS+2*Delta+3*U-J),
    (2,2):
        (4*eS+2*Delta+6*U-2*J,)
}


def test_solve():
    
    espace, egs = build_espace(H, dict(U=V,Jx=Jx,Jp=Jp))

    for key, vals in eigvals.items():
        np.testing.assert_allclose(vals, espace[key].eigvals) 
