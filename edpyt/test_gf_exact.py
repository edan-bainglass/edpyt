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
    gf = build_gf_exact(H, V, beta=beta, mu=mu)

    eta = 1e-3
    energies = 10*np.random.random(20) - 5

    # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
    expected = 0.5*(1/(energies+1.j*eta+0.5*U)+1/(energies+1.j*eta-0.5*U))
    assert np.allclose(expected, gf(energies, eta))


def test_gf_exact_plot():

    from matplotlib import pyplot as plt

    ed = 0.5
    t = 0.2
    beta = 100

    H = np.array([
        [ed,-t],
        [-t,ed]
    ])

    eta = 1e-3
    energies = np.arange(-2,2,1e-3)

    for iU, U in enumerate(np.arange(0,10*t,0.5)):

        mu = ed + U/2.

        V = np.array([
            [U,0],
            [0,U]
        ])

        gf = build_gf_exact(H, V, beta=beta, mu=mu)

        # https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
        plt.plot(energies, -1/np.pi*gf(energies, eta).imag + iU*10)

    plt.savefig('gf.png')
    plt.close()
