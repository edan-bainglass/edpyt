import numpy as np
from scipy import linalg as la

# Physical quantities
from scipy.constants import e, k


'''References:

i) http://www1.spms.ntu.edu.sg/~ydchong/teaching/08_contour_integration.pdf

ii) Efficient Integration of Green's functions: Ver. 1.1

'''

def zero_fermi(nzp):
    '''Compute poles (zp) and residues (Rp) of fermi function.'''
    N = nzp
    M = 2*N

    A = np.zeros((2*M,2*M))
    B = np.zeros((2*M,2*M))

    zp = np.zeros(2+M)
    Rp = np.zeros(2+M)

    for i in range(1,M+1):
        B[i,i] = (2*i-1)

    for i in range(1,M):
        A[i,i+1] = -0.5
        A[i+1,i] = -0.5

    a = np.zeros(M*M)
    b = np.zeros(M*M)

    for i in range(M):
        for j in range(M):
            a[j*M+i] = A[i+1,j+1]
            b[j*M+i] = B[i+1,j+1]

    a.shape = (M,M)
    b.shape = (M,M)

    eigvals, eigvecs = la.eigh(a,b)

    zp[:M] = eigvals

    for i in range(M,0,-1):
        zp[i] = zp[i-1]

    for i in range(1,M+1):
        zp[i] = 1.0/zp[i]

    a = eigvecs.T.flatten()

    for i in range(0,M):
        Rp[i+1] = -a[i*M]*a[i*M]*zp[i+1]*zp[i+1]*0.250

    zp = -zp[1:N+1]
    Rp = Rp[1:N+1]

    return zp, Rp


def integrate_gf(gf, mu=0, T=300, nzp=100, R=1e10):
    """Integrate the green's function up to the chemical potential.
    
    NOTE :: do not include the spin degeneracy factor, i.e.
    
    assuming gf describes both up and down spins:

            n   + n   = 2 * integrate_gf(gf)
            up    dw
    """
    _gf = lambda z: np.diagonal(np.atleast_2d(gf(z)))
    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1/(k_B*T)
    a_p = mu + 1j*zp/beta

    R = 1e10
    mu_0 = 1j*R*_gf(1.j*R)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += _gf(a_p[i]) * Rp[i]
    mu_1 *= -1j*4/beta

    rho = np.real(mu_0) + np.imag(mu_1)

    return rho/2.


def matsubara_sum(gf, nmats=100, beta=10.):
    wn = (2*np.arange(nmats)+1)*np.pi/beta
    