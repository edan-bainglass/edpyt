import numpy as np

from gf_exact import (
    build_gf_exact,
)

"""Hubbard dimer atomic limit.

"""

t = 0.
ed = 0.5
U = 1.

beta = 0.
mu = ed + U/2.

H = np.array([
    [ed,-t],
    [-t,ed]
])

V = np.array([
    [U,0],
    [0,U]
])

def test_build_gf_exact():
    # Test against atomic limit
    gf = build_gf_exact(H, V, beta=0., mu=mu)

    eta = 1e-3
    energies = 10*np.random.random(20) - 5

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    expected = 0.5*(1/(energies+1.j*eta+0.5*U)+1/(energies+1.j*eta-0.5*U))
    assert np.allclose(expected, gf(energies, eta))
