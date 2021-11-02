from collections import namedtuple
import numpy as np

from edpyt.dedlib import (
    RandomSampler,
    Gf0,
    Gfimp,
    build_moam,
    build_siam, get_evecs_occupation,
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
    rs = RandomSampler(lorentz, [-5.,5.], n)
    gf0 = Gf0(rs)
    gfimp = Gfimp(n)
    gf0.sample()
    gfimp.fit(gf0)
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
    build_siam(H, gfimp)
    V[0,0] = 0.
    neig = np.ones((n+1)*(n+1),int)
    espace, egs = build_espace(H, V, neig)
    for (nup, ndw), sct in espace.items():
        # try:
        #     evec = sct.eigvecs[:,0]
        # except IndexError:
        #     evec = sct.eigvecs
        N0 = 0.
        for pos in range(n):
            N0 += get_evecs_occupation(sct.eigvecs,
                                       np.ones((1,)),
                                       sct.states.up,
                                       sct.states.dw,
                                       pos)
        np.testing.assert_allclose(N0, nup+ndw)

    # Test non-interacting Green's function.
    screen_espace(espace, egs, beta=1e4)
    gf = build_gf_lanczos(H, V, espace, 0.)
    np.testing.assert_allclose(gf(energies,0.02),gf0(energies+1.j*0.02))


def test_moam():
    gf = namedtuple('gf',['ek','vk2','e0'])
    nbath = 2
    nimp = 3
    n = (nbath+1)*nimp
    H = np.zeros((n,n))
    V = H.copy()
    ek = np.ones(nbath)
    vk2 = ek*4.
    gfimp = [gf(ek+i,vk2**i,i) for i in range(1,nimp+1)]
    build_moam(H, gfimp)
    np.testing.assert_allclose(H,  [[ 1., 0., 0.,-2.,-2., 0., 0., 0., 0.],
                                    [ 0., 2., 0., 0., 0.,-4.,-4., 0., 0.],
                                    [ 0., 0., 3., 0., 0., 0., 0.,-8.,-8.],
                                    [-2., 0., 0., 2., 0., 0., 0., 0., 0.],
                                    [-2., 0., 0., 0., 2., 0., 0., 0., 0.],
                                    [ 0.,-4., 0., 0., 0., 3., 0., 0., 0.],
                                    [ 0.,-4., 0., 0., 0., 0., 3., 0., 0.],
                                    [ 0., 0.,-8., 0., 0., 0., 0., 4., 0.],
                                    [ 0., 0.,-8., 0., 0., 0., 0., 0., 4.]])