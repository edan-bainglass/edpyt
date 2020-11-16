import numpy as np

from fit import (
    fit_hybrid
)

# def test_fit_hybrid():
#     # Analytic expression of the Hilbert transform of the Bethe lattice
#     hybrid_true = lambda z: 2*(z-np.sqrt(z**2-1))
#
#     n = 10
#     nmats = 100
#     beta = 10
#
#     hybrid_disc = fit_hybrid(hybrid_true, n, nmats, beta)
#
#     energies = np.arange(-5,5,0.1)
#     eta = 0.25
#
#     np.testing.assert_allclose(
#         hybrid_disc(energies,eta),
#         hybrid_true(energies+1.j*eta)
#     )
