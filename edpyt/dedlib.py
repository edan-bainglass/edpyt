import numpy as np
from numba import njit

# Fig gfimp
from scipy.optimize import fsolve

# Smooth
from scipy.interpolate import interp1d
from scipy import fftpack

from edpyt.espace import build_espace
from edpyt.gf_lanczos import build_gf_lanczos

# Sampling
from numba import njit
import random as _random

# Reference:

#  Kondo physics of the Anderson impurity model by Distributional ExactDiagonalization.
#  S. Motahari,  R.  Requist,  and  D.  Jacob


@njit()
def cumsum(a):
    for i in range(1,a.size):
        a[i] += a[i-1]


def get_random_sampler(p, lims, n=4, rng=_random):
    """Get random sampler for probability p.

    Args:
        p : (callable) probability distribution.
        lims : bounds for sampling.
        n : # of samples.
        rng : random number generator.

    Returns:
        random_sampler : (callable)
            Sample n randoms uniformely in range [lims[0],ilms[1]].

    Example:
        p = lambda x: x
        x = np.linspace(0,1,1001,endpoint=True)
        sampler = get_random_sampler(p, x, n=4)
        sampler()
        Out : array([0.68814168, 0.84116257, 0.95736033, 0.66379715])

    """
    eners = np.linspace(lims[0],lims[1],int(np.diff(lims)*1e5+1),endpoint=True)
    cdf = p(eners)
    cdf /= cdf.sum()
    cumsum(cdf)
    poles = np.empty(n)
    def sampler():
        for i in range(n):
            poles[i] = eners[np.searchsorted(cdf, rng.random())]
        poles.sort()
        return poles
    return sampler


def build_gf0(poles):
    """Build non-interacting Green's function.

    """
    #           __ n
    #  0       \                1/n
    # G (z)                  ---------
    #          /__ i = 0       z - poles
    #                                   i
    n = poles.size
    def gf0(z):
        z = np.array(z, ndmin=1)
        return 1/n * np.reciprocal(z[:,None]-poles[None,:]).sum(1)
    gf0.poles = poles
    gf0.n = poles.size
    gf0.derivative = lambda z: np.sum(-1/(z-gf0.poles)**2)/gf0.n
    return gf0


def build_gfimp(gf0):
    """Fit SAIM model.

    Returns:
        G(z) : (callable) impurity Green's function.
        - G.ek = onsite bath energies.
        - G.vk2 = imp-bath square hoppings.
        - G.e0 = impurity energy level.
    """
    #
    #  0      !     1
    # G (z)   =  ----------  __ n-1           2
    #            z - e0 -    \               Vk
    #                                     -------
    #                        /__k = 0      z - ek
    poles = gf0.poles
    n = gf0.n
    ek = np.zeros(n-1)
    vk2 = np.zeros(n-1)
    for i in range(n-1):
        ek[i] = fsolve(gf0, (poles[i+1]+poles[i])/2)
        vk2[i] = -1/gf0.derivative(ek[i])
    e0 = poles.mean()
    def gfimp(z):
        z = np.array(z, ndmin=1)
        delta = lambda z: vk2[None,:] * np.reciprocal(z[:,None]-ek[None,:])
        return np.reciprocal(z-e0-delta(z).sum(1))
    gfimp.e0 = e0
    gfimp.ek = ek
    gfimp.vk2 = vk2
    return gfimp


@njit()
def get_occupation(vector, states_up, states_dw, n):
    """Count particles in eigen-state vector (size=dup x dwn).

    """
    N = 0.
    occps = vector**2
    d = states_up.size*states_dw.size
    dup = states_up.size
    dwn = states_dw.size

    for i in range(d):
        iup = i%dup
        idw = i//dup
        sup = states_up[iup]
        sdw = states_dw[idw]
        for j in range(0,n):
            if (sup>>j)&np.uint32(1):
                N += occps[i]
            if (sdw>>j)&np.uint32(1):
                N += occps[i]
    return N


def keep_gs(espace, egs):
    """Keep sectors containing ground state.

    """
    delete = []
    for (nup, ndw), sct in espace.items():
        diff = np.abs(sct.eigvals[0]-egs) < 1e-7
        if ~diff.any():
            delete.append((nup,ndw))
    for k in delete:
        espace.pop(k)


def build_siam(H, V, U, gfimp):
    """Build single Anderson Impurity model.

    """
    vk = np.sqrt(gfimp.vk2)
    n = H.shape[0]
    H[1:,0] = H[0,1:] = - vk
    H.flat[(n+1)::(n+1)] = gfimp.ek
    H[0,0] = gfimp.e0
    V[0,0] = U


def ded_solve(dos, z, sigma=None, sigma0=None, n=4, N=int(1e3), U=3., rng=np.random):
    """Solve SIAM with DED.

    Args:
        dos : (callable)
        z : np.array(,dtype=complex)
        sigma : np.array(,dtype=complex) or None
        sigma0 : Re(sigma[energy_fermi])
        n : # of poles
        N : # of steps that fulfill DED condition **
        rng : (np.random.Generator) use to improve indipendent
            samplings of the poles for parallel tasks.

    Returns:
        sigma/None : if sigma input is None or not, respectively.


    ** DED condition : N==N0
    i) N # of GS particles with U!=0
    i) N0 # of GS particles with U==0

    """
    _sigma = sigma or np.zeros_like(z)
    _sigma0 = sigma0 or U/2.
    H = np.zeros((n,n))
    V = np.zeros((n,n))
    neig = np.ones((n+1)*(n+1)) * 1
    rs = get_random_sampler(dos, [z.real[0],z.real[-1]], n, rng)
    for _ in range(N):
        found = False
        while not found:
            poles = rs()
            gf0 = build_gf0(poles)
            gfimp = build_gfimp(gf0)
            build_siam(H, V, 0., gfimp)
            espace, egs = build_espace(H, V, neig)
            keep_gs(espace, egs)
            sct = next(v for v in espace.values())
            if sct.eigvecs.ndim < 2: continue
            evec = sct.eigvecs[:,0]
            N0 = get_occupation(evec,sct.states.up,sct.states.dw,n)
            V[0,0] = U
            H[0,0] -= _sigma0
            espace, egs = build_espace(H, V, neig)
            keep_gs(espace, egs)
            sct = next(v for v in espace.values())
            evec = sct.eigvecs[:,0]
            Nv = get_occupation(evec,sct.states.up,sct.states.dw,n)
            if np.allclose(Nv,N0):
                gf = build_gf_lanczos(H, V, espace, 0.)
                _sigma += np.reciprocal(gf0(z))-np.reciprocal(gf(z.real,z.imag))
                found = True
    if sigma is None:
        return _sigma


def smooth(energies, sigma, cutoff=2):
    """Helper function to interpolate interacting self-energy.

    Example usage:
        sigma_interp = smooth(energies, sigma)
        sigma0 = sigma_interp(0.).real

    """
    # Interpolate on regular meshgrid
    sig_interp = interp1d(energies, sigma)
    ne_interp = (energies.max()-energies.min())/1e-2 + 1
    energies_interp = np.linspace(energies.min(),energies.max(),int(ne_interp),endpoint=True)
    sig = sig_interp(energies_interp)
    #
    sample_freq = fftpack.fftfreq(sig.size, d=energies_interp[1]-energies_interp[0])
    # Smooth Imag
    sig_imag_fft = fftpack.fft(sig.imag)
    sig_imag_fft[np.abs(sample_freq) > cutoff] = 0
    filtered_imag_sig = fftpack.ifft(sig_imag_fft).real
    # Smooth Real
    sig_real_fft = fftpack.fft(sig.real)
    sig_real_fft[np.abs(sample_freq) > cutoff] = 0
    filtered_real_sig = fftpack.ifft(sig_real_fft).real
    # Merge and Interpolate
    smooth_sig = filtered_real_sig + 1.j*filtered_imag_sig
    smooth_sig_interp = interp1d(energies_interp, smooth_sig, bounds_error=False, fill_value=0.)
    return smooth_sig_interp
