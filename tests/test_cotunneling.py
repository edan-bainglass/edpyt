import numpy as np
from collections import defaultdict

# from edpyt.espace import Sector, build_empty_sector
# from edpyt.cotunneling import project


# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.77.045329


def test_project():
    pass
    A = np.array([
        [1],
        [1]
    ])
    eup = 1.
    edw = 2.
    U = 3.
    n = 1
    espace = defaultdict(Sector)
    eigvals = [0.,edw,eup,eup+edw+U]
    for i, (nup, ndw) in enumerate(np.ndindex((n+1, n+1))):
        sct = build_empty_sector(n, nup, ndw)
        sct.eigvecs = np.array(1., ndmin=2)
        sct.eigvals = np.array(eigvals[i], ndmin=1)
        espace[(nup,ndw)] = sct
    egs = np.min(eigvals)

    # UP spin
    sigma, de = project(A, 0, 1, 1, 0, espace)
    np.testing.assert_allclose(sigma[(0,0)].Gf2h.E, eup)
    np.testing.assert_allclose(sigma[(1,1)].Gf2e.E, edw+U)
    np.testing.assert_allclose(sigma[(0,1)].Gf2e.E, edw+U)
    np.testing.assert_allclose(sigma[(0,1)].Gf2h.E, eup)
    np.testing.assert_allclose(sigma[(0,1)](0.), (1/-eup - 1/-(edw+U))**2)

    # DW spin
    sigma, de = project(A, 0, 1, 0, 1, espace)
    np.testing.assert_allclose(sigma[(1,1)].Gf2h.E, edw)
    np.testing.assert_allclose(sigma[(0,0)].Gf2e.E, eup+U)
    np.testing.assert_allclose(sigma[(1,0)].Gf2e.E, eup+U)
    np.testing.assert_allclose(sigma[(1,0)].Gf2h.E, edw)

if __name__ == '__main__':
    test_project()