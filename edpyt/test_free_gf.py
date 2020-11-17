import numpy as np

from gf_lanczos import (
    build_gf_lanczos
)

from gf_exact import (
    build_gf_exact
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


"""Free particle Green's function.

"""

n = 5
nup = 2
ndw = 3

beta = 10.

t = 1/np.sqrt(n-1)
delta = 1/(n-1)
ed = np.array([-1 + i*delta for i in range(n-1)])
# ed = np.ones(n-1)*0.1

H = np.zeros((n,n))
H[0,1:] = H[1:,0] = -t
for i in range(1,n):
    H[i,i] = ed[i-1]

V = np.zeros_like(H)

def test_free_gf_lanczos():
    from matplotlib import pyplot as plt

    from shared import params
    # params['mu'] = ed

    neig_sector = np.zeros((n+1)*(n+1), int)

    neig_sector[
        get_sector_index(nup, ndw, n)
        ] = 5
    neig_sector[
        get_sector_index(ndw, nup, n)
        ] = 5

    eimp = - 0.

    H[0,0] = eimp
    espace, egs = build_espace(H, V, neig_sector)
    gf = build_gf_lanczos(H, V, espace, beta=0., mu=0.)

    eta = 0.25
    energies = np.arange(-10,10,1e-3)

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    z = energies + 1.j*eta
    hybr = (t**2 / (z[:,None]-ed[None,:])).sum(1)
    gf_expected = 1/(z-eimp-hybr)
    gf_computed = gf(energies, eta)

    plt.plot(energies,-1/np.pi*gf_expected.imag,label='expected')
    plt.plot(energies,-1/np.pi*gf_computed.imag,label='computed')
    plt.legend()
    plt.savefig('g0_lanczos.png')
    plt.close()

    np.testing.assert_allclose(gf_computed,gf_expected,0.5)
    # assert np.allclose(gf_computed,gf_expected,atol=1.)


def test_free_gf_exact():
    from matplotlib import pyplot as plt

    # from shared import params
    # params['mu'] = ed
    eimp = - 2.
    H[0,0] = eimp

    gf = build_gf_exact(H, V, beta=beta, mu=0.)

    eta = 0.25
    energies = np.arange(-10,10,1e-3)

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    z = energies + 1.j*eta
    hybr = (t**2 / (z[:,None]-ed[None,:])).sum(1)
    gf_expected = 1/(z-eimp-hybr)
    gf_computed = gf(energies, eta)

    plt.plot(energies,-1/np.pi*gf_expected.imag,label='expected')
    plt.plot(energies,-1/np.pi*gf_computed.imag,label='computed')
    plt.legend()
    plt.savefig('g0_exact.png')
    plt.close()

    assert np.allclose(gf_computed,gf_expected,0.5)
