import numpy as np

from edpyt.gf_lanczos import (
    build_gf_lanczos
)

from edpyt.gf_exact import (
    build_gf_exact
)

from edpyt.espace import (
    build_espace
)

from edpyt.lookup import (
    get_sector_index
)

from edpyt.sector import (
    get_sector_dim
)


"""Free particle Green's function.

"""

n = 5
nup = 2
ndw = 3

beta = 0.                # Zero Temperature
eimp = - 2.              # Impurity level

t = 1/np.sqrt(n-1)
delta = 1/(n-1)
ed = np.array([-1 + i*delta for i in range(n-1)])

H = np.zeros((n,n))
H[0,1:] = H[1:,0] = -t   # Hoppings
for i in range(1,n):     # Bath energies
    H[i,i] = ed[i-1]
H[0,0] = eimp            # Impurity

V = np.zeros_like(H)     # Free particle

def test_gf_lanczos_free():
    #
    #
    #
    #                   1
    # G   (z)  =  -------------
    #  00            z -  e   - Delta(z)
    #                      imp
    #
    #
    #                ____ nbath        2
    #               \            |b(i)|
    #  Delta(z) =    \          ---------
    #                /           z - a(i)
    #               /____ i=0
    from matplotlib import pyplot as plt
    from edpyt.shared import params
    params['hfmode'] = False
    params['mu'] = 0.

    neig_sector = np.zeros((n+1)*(n+1), int)

    neig_sector[
        get_sector_index(nup, ndw, n)
        ] = 5
    neig_sector[
        get_sector_index(ndw, nup, n)
        ] = 5

    espace, egs = build_espace(H, V, neig_sector)
    gf = build_gf_lanczos(H, V, espace, beta=0., mu=0.)

    eta = 0.25
    energies = np.arange(-10,10,1e-3)

    z = energies + 1.j*eta
    hybr = (t**2 / (z[:,None]-ed[None,:])).sum(1)
    gf_expected = 1/(z-eimp-hybr)
    gf_computed = gf(energies, eta)

    plt.plot(energies,-1/np.pi*gf_expected.imag,label='expected')
    plt.plot(energies,-1/np.pi*gf_computed.imag,label='computed')
    plt.legend()
    plt.savefig('g0_lanczos.png')
    plt.close()

    np.testing.assert_allclose(gf_computed,gf_expected,1e-3)


if __name__ == '__main__':
    test_gf_lanczos_free()
