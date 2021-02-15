import numpy as np

from edpyt.espace import (build_espace, build_non_interacting_espace)

n = 4 
ek = np.random.random(n)
H = np.diag(ek)
V = np.zeros_like(H)

def test():
    espace, egs = build_espace(H, V)
    espace0, egs0 = build_non_interacting_espace(ek)

    np.testing.assert_allclose(egs, egs0)
    for sct, sct0 in zip(espace.values(), espace0.values()):
        np.testing.assert_allclose(sct.eigvals, sct0.eigvals)
