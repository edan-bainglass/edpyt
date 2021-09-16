import numpy as np
from edpyt.fit import cityblock, fit_hybrid

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


def test_cityblock():
    n = 100
    rnd = np.random.random
    a = rnd(n) + 1.j * rnd(n)
    b = rnd(n) + 1.j * rnd(n)
    w = np.ones(n)
    
    np.testing.assert_allclose(np.sqrt((w*(a-b)*(a-b).conj()).sum()),np.linalg.norm(a-b))
    
    w = rnd(n)
    np.testing.assert_allclose(np.sqrt((w*(a-b)*(a-b).conj()).sum()),cityblock(a,b,w))
    

# def test_fit():    
#     beta = 70.
#     nmats = 3000
#     hybrid_true = lambda z: 2*(z-np.sqrt(z**2-1))
#     z = 1.j*(2*np.arange(3000)+1)*np.pi/70.
#     vals_true = hybrid_true(z)
#     vals_true = np.tile(vals_true, (4,1))
#     popt, fopt = fit_hybrid(vals_true)
#     print(fopt)
