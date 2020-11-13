import numpy as np

from gf_lanczos import (
    build_gf_lanczos
)

from espace import (
    build_espace
)

from lookup import (
    get_sector_index
)

from sector import (
    get_sector_dim
)

"""Hubbard dimer atomic limit.

"""

t = 0.
ed = 0.5
U = 1.

beta = np.inf # Zero T limit
mu = ed + U/2.

H = np.array([
    [ed,-t],
    [-t,ed]
])

V = np.array([
    [U,0],
    [0,U]
])

def test_build_gf_lanczos():
    # Test against atomic limit
    n = H.shape[0]
    neig_sector = np.zeros((n+1)*(n+1), int)
    neig_sector[
        get_sector_index(n//2, n//2, n)] = get_sector_dim(n, n//2) ** 2
    espace, egs = build_espace(H, V, neig_sector)
    gf = build_gf_lanczos(H, V, espace, beta=0., mu=mu)

    eta = 1e-3
    energies = 10*np.random.random(20) - 5

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    expected = 0.5*(1/(energies+1.j*eta+0.5*U)+1/(energies+1.j*eta-0.5*U))
    assert np.allclose(expected, gf(energies, eta))
