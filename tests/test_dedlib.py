import numpy as np

from edpyt.dedlib import (
    get_random_sampler,
    build_gf0,
    build_gfimp,
    build_siam,
    get_occupation,
)

from edpyt.espace import (
    build_espace,
    screen_espace
)

from edpyt.gf_lanczos import (
    build_gf_lanczos
)

lorentz = lambda z: 1/np.pi * (0.15)/(z**2+0.15**2) #Gamma==0.3

def _setup(n):
    rs = get_random_sampler(lorentz, [-5.,5.], n)
    poles = rs()
    gf0 = build_gf0(poles)
    gfimp = build_gfimp(gf0)
    return gf0, gfimp

def test_dedlib():
    n = 6
    H = np.zeros((n,n))
    V = np.zeros((n,n))
    gf0, gfimp = _setup(n)

    # Test fit
    energies = 10. * np.random.random(30) - 5.
    np.testing.assert_allclose(gf0(energies), gfimp(energies))

    # Test occupation
    build_siam(H, V, 0., gfimp)
    neig = np.ones((n+1)*(n+1),int)
    espace, egs = build_espace(H, V, neig)
    for (nup, ndw), sct in espace.items():
        try:
            evec = sct.eigvecs[:,0]
        except IndexError:
            evec = sct.eigvecs
        N0 = get_occupation(evec,sct.states.up,sct.states.dw,n)
        np.testing.assert_allclose(N0, nup+ndw)

    # Test non-interacting Green's function.
    screen_espace(espace, egs, beta=1e4)
    gf = build_gf_lanczos(H, V, espace, 0.)
    np.testing.assert_allclose(gf(energies,0.02),gf0(energies+1.j*0.02))
