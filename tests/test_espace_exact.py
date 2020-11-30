import numpy as np

from edpyt.espace import (
    build_espace,
)

"""Hubbard dimer.

"""


t = 1.
U = 1.
ed = 0.5

H = np.array([
    [ed,-t],
    [-t,ed]
])

V = np.array([
    [U,0],
    [0,U]
])

# https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
eigvals = {
    (0,0):[0.],
    (1,0):[ed-t,ed+t],
    (0,1):[ed-t,ed+t],
    (1,1):[2.*ed+0.5*(U-np.sqrt(U**2.+16.*t**2.)),
           2.*ed,2.*ed+U,
           2.*ed+0.5*(U+np.sqrt(U**2.+16.*t**2.))],
    (0,2):[2.*ed,2.*ed],
    (2,0):[2.*ed,2.*ed],
    (2,1):[3.*ed+U-t,3.*ed+U+t],
    (1,2):[3.*ed+U-t,3.*ed+U+t],
    (2,2):[4.*ed+2*U]
}
eigvals = {key:np.array(eig) for key, eig in eigvals.items()}
# Ground state
EG0 = 2.*ed+0.5*(U-np.sqrt(U**2.+16.*t**2.))


def test_solve():
    espace, egs = build_espace(H, V)

    # Eigenspace
    for (nup, ndw), sct in espace.items():
        assert np.allclose(
            sct.eigvals,eigvals[(nup,ndw)])

    # Ground state
    assert np.allclose(egs, EG0)
