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

H_ = lambda ed, t: np.array([
    [ed,-t],
    [-t,ed]
])

V_ = lambda U: np.array([
    [U,0],
    [0,U]
])

def test_build_gf_exact():
    # Test against atomic limit
    H = H_(ed,t)
    V = V_(U)
    gf = build_gf_exact(H, V, beta=beta, mu=mu)

    eta = 1e-3
    energies = 10*np.random.random(20) - 5

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    expected = 0.5*(1/(energies+1.j*eta+0.5*U)+1/(energies+1.j*eta-0.5*U))
    assert np.allclose(expected, gf(energies, eta))

def test_phs():
    # Test particle hole symmetry
    from shared import params

    t = 0.1
    H = H_(ed,t)
    V = V_(U)

    params['hfmode'] = False
    mu = ed + U/2.
    gf = build_gf_exact(H, V, beta=beta, mu=mu)

    params['hfmode'] = True # U(n-0.5)(n-0.5)
    mu = ed
    gf_phs = build_gf_exact(H, V, beta=beta, mu=mu)

    eta = 1e-3
    energies = 10*np.random.random(20) - 5

    #https://www.cond-mat.de/events/correl16/manuscripts/scalettar.pdf
    assert np.allclose(gf_phs(energies, eta), gf(energies, eta))
