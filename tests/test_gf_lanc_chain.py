import numpy as np
from matplotlib import pyplot as plt

from edpyt.gf_lanczos import (
    build_gf_lanczos
)

from edpyt.espace import (
    build_espace
)

from edpyt.lookup import (
    get_sector_index
)

n = 12
nup = ndw = 6 # Half-filling

t = 0.
U = 8.

H = np.diag([-t]*(n-1),k=1) + np.diag([-t]*(n-1),k=-1)
V = np.diag([U]*n)


def test_chain_atomic_limit():
    #                     _                              _
    #                1   |       0.5             0.5      |
    #  G  (z) =   - ---  |     --------   +     --------  |
    #   00           pi  |_    z + U/2          z - U/2  _|
    #
    from edpyt.shared import params
    params['mu'] = 0.
    params['hfmode'] = True

    neig_sector = np.zeros((n+1)*(n+1))
    neig_sector[get_sector_index(n, n//2, n//2)] = 1

    espace, egs = build_espace(H, V, neig_sector)
    # Continued Fraction representation
    gf = build_gf_lanczos(H, V, espace, beta=0.)
    # Spectral representation
    gf_sp = build_gf_lanczos(H, V, espace, beta=0., repr='sp')

    energies = np.arange(-20,20,1e-3)
    eta = 0.02
    z = energies + 1.j*eta

    computed = -1/np.pi*gf(energies, eta).imag
    computed_sp = -1/np.pi*gf_sp(energies, eta).imag
    expected = -1/np.pi*(0.5/(z+U/2) + 0.5/(z-U/2)).imag

    np.testing.assert_allclose(expected, computed, rtol=1e-2)
    np.testing.assert_allclose(expected, computed_sp, rtol=1e-2)

    plt.plot(energies[::5], expected[::5], 'o', label='expected')
    plt.plot(energies, computed, label='computed')
    plt.legend()
    plt.savefig('g0_atomic')
    plt.close()

if __name__ == '__main__':
    test_chain_atomic_limit()
