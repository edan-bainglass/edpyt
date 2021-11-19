from collections import defaultdict
import numpy as np
from numba import njit

# Fig gfimp
from scipy.optimize import fsolve

# Smooth
from scipy.interpolate import interp1d
from scipy import fftpack
from scipy.constants import physical_constants

from edpyt.gf_lanczos import build_gf_lanczos

kB = physical_constants['Boltzmann constant in eV/K'][0]

from edpyt.espace import (build_espace, screen_espace, adjust_neigsector, 
    build_non_interacting_espace)
from edpyt.gf_exact import build_gf_exact
from edpyt.operators import check_full

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


# https://sciencehouse.wordpress.com/2015/06/20/sampling-from-a-probability-distribution/
class RandomSampler:
    '''Random sampler for custum probability. 
    
    '''
    def __init__(self, p, lims, n=4, rng=_random, nsamples=1e6):
        '''
        Args:
            p : (callable) probability distribution.
            lims : bounds for sampling.
            n : # of samples.
            rng : random number generator.
            nsamples : # of cdf samples.
        '''
        self.eners = np.linspace(lims[0],lims[1],int(np.diff(lims)*nsamples+1),
                                 endpoint=True)
        cdf = p(self.eners)
        cdf /= cdf.sum()
        cumsum(cdf)
        self.cdf = cdf
        self.poles = np.empty(n)
        self.rng = rng
    
    def __call__(self):
        '''Sample n randoms uniformely distributed in range [lims[0],lims[1]].
        
        '''
        n = self.poles.size
        for i in range(n):
            self.poles[i] = self.eners[np.searchsorted(self.cdf, self.rng.random())]
        self.poles.sort()


class Gf0:
    """Non-interacting Green's function.

    """
    #           __ n
    #  0       \                1/n
    # G (z)                  ---------
    #          /__ i = 0       z - poles
    #                                   i
    def __init__(self, rs):
        '''
        Args:
            rs : RandomSampler instance.
        '''
        self.rs = rs
        self.poles = self.rs.poles
        self.n = self.poles.size

    def sample(self):
        self.rs()

    def __call__(self, z):
        return 1/self.n * np.reciprocal(z[:,None]-self.poles[None,:]).sum(1)
    
    def derivative(self, z0):
        return np.sum(-1/(z0-self.poles)**2)/self.n


class Gfimp:
    """Non-interacting Green's function of SIAM model.

    """
    #
    #  SIAM         1
    # G (z)   =  ----------  __ n-1           2
    #            z - e0 -    \               Vk
    #                                     -------
    #                        /__k = 0      z - ek
    def __init__(self, n):
        self.ek = np.empty(n-1)
        self.vk2 = np.empty(n-1)
        self.e0 = 0.

    def fit(self, gf0):
        '''Fit non-interacting Green's function.

             0    !  SIAM
            G (z) = G   (z)
        '''
        poles = gf0.poles
        n = poles.size
        for i in range(n-1):
            self.ek[i] = fsolve(gf0, (poles[i+1]+poles[i])/2)
            self.vk2[i] = -1/gf0.derivative(self.ek[i])
        self.e0 = poles.mean()

    def delta(self, z):
        return self.vk2[None,:] * np.reciprocal(z[:,None]-self.ek[None,:])

    def __call__(self, z):
        return np.reciprocal(z-self.e0-self.delta(z).sum(1))


@njit()
def get_occupation(vector, states, pos):
    """Count particles in eigen-state vector (size=dup x dwn).

    """
    N = 0.
    dup = states_up.size
    dwn = states_dw.size
    occps = (vector**2).reshape(dwn, dup)

    for iup in range(dup):
        sup = states_up[iup]
        if check_full(sup, pos):
            N += occps[:,iup].sum()

    for idw in range(dwn):
        sdw = states_dw[idw]
        if check_full(sdw, pos):
            N += occps[idw,:].sum()

    return N


def get_entropy(gfimp, espace, egs, beta):
    """Compute impurity entropy.
    
    Ref :: https://d-nb.info/1147380651/34
    """
    #     ___
    #     \                   < H >
    # S =     kB ln(Z(T)) +  -------   
    #     /__                   T
    espace_bath, egs_bath = build_non_interacting_espace(gfimp.ek)
    E, Z = _get_entropy_params(espace, egs, beta)
    E_bath, Z_bath = _get_entropy_params(espace_bath, egs_bath, beta)

    T = 1/(kB*beta) # beta = 1/(kB*T)
    S = _calc_entropy(Z, E, T)
    S_bath = _calc_entropy(Z_bath, E_bath, T)
    return S - S_bath

_calc_entropy = lambda Z, E, T: kB*np.log(Z) + E/T
_calc_entropy.__doc__ = """Helper function to assemble Entropy."""

def _get_entropy_params(espace, egs, beta):
    """Helper functionto compute < H > and Z.
    
    """
    E = 0.; Z = 0. 
    for sct in espace.values():
        eigvals = sct.eigvals - egs
        exps = np.exp(-beta*eigvals)
        Z += exps.sum()
        E += (eigvals*exps).sum()
    E /= Z
    return E, Z        


def build_siam(H, V, U, gfimp):
    """Build single Anderson Impurity model.

    """
    vk = np.sqrt(gfimp.vk2)
    n = H.shape[0]
    H[1:,0] = H[0,1:] = - vk
    H.flat[(n+1)::(n+1)] = gfimp.ek
    H[0,0] = gfimp.e0
    V[0,0] = U


def siam_solve(dos, z, sigma=None, sigma0=None, n=4, 
              N=int(1e3), U=3., beta=1e6, rng=_random):
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
    return_sigma = False
    if sigma is None:
        sigma = np.zeros_like(z)
        return_sigma = True
    if sigma0 is None: sigma0 = U/2. # chemical potential
    imp_occp0 = 0.                   # non-interacting impurity occupation
    imp_occp1 = 0.                   # interacting impurity occupation
    imp_entropy = 0.                 # impurity entropy
    H = np.zeros((n,n))
    V = np.zeros((n,n))
    rs = RandomSampler(dos, [z.real[0],z.real[-1]], n, rng)
    gf0 = Gf0(rs)
    gfimp = Gfimp(n)
    neig1 = None #np.ones((n+1)*(n+1),int) * 3
    neig0 = None #np.ones((n+1)*(n+1),int) * 3
    for _ in range(N):
        found = False
        while not found:
            gf0.sample()
            gfimp.fit(gf0)
            build_siam(H, V, 0., gfimp)
            espace, egs = build_espace(H, V, neig0)
            # screen_espace(espace, egs, beta)
            # adjust_neigsector(espace, neig0, n)
            N0, sct = next((k,v) for k,v in espace.items() if abs(v.eigvals[0]-egs)<1e-7)
            evec = sct.eigvecs[:,0]
            occp0 = get_occupation(evec,sct.states.up,sct.states.dw,0)
            V[0,0] = U
            H[0,0] -= sigma0
            espace, egs = build_espace(H, V, neig1)
            # screen_espace(espace, egs, beta)
            # adjust_neigsector(espace, neig1, n)
            N1, sct = next((k,v) for k,v in espace.items() if abs(v.eigvals[0]-egs)<1e-7)
            if np.allclose(N1,N0):
                gf = build_gf_exact(H,V,espace,beta,egs)
                sigma += np.reciprocal(gf0(z))-np.reciprocal(gf(z.real,z.imag))
                evec = sct.eigvecs[:,0]
                occp1 = get_occupation(evec,sct.states.up,sct.states.dw,0)
                imp_occp0 += occp0
                imp_occp1 += occp1
                imp_entropy += get_entropy(gfimp,espace,egs,beta)
                found = True
    sigma /= N
    imp_occp0 /= N
    imp_occp1 /= N
    imp_entropy /= (N*kB)
    if return_sigma:
            return sigma, imp_occp0, imp_occp1, imp_entropy
    return imp_occp0, imp_occp1, imp_entropy


def build_moam(H, gfimp):
    """Build multi orbital Anderson model.

    """
    #
    #  | e'         vk'0   0     |
    #  |       e''     0   vk''0 |
    #  |vk'0    0   ek'0         |
    #  |  0   vk''0        ek''0 |
    nimp = len(gfimp)
    n = H.shape[-1]
    for i, gf in enumerate(gfimp):
        vk = np.sqrt(gf.vk2)
        nbath = vk.size
        bl = i*nbath+nimp
        bh = bl + nbath
        H[bl:bh,i] = H[i,bl:bh] = - vk
        dl = (nimp+i*nbath)*(n+1)
        dh = dl + nbath*(n+1)
        H.flat[dl:dh:(n+1)] = gf.ek
        H[i,i] = gf.e0


def moam_solve(dos, z, sigma=None, sigma0=None, n=4, 
              N=int(1e3), U=3., beta=1e6, rng=_random):
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
    nimp = len(dos)
    return_sigma = False
    if sigma is None:
        sigma = np.zeros((nimp,z.size),complex)
        return_sigma = True
    if sigma0 is None: sigma0 = U/2. # chemical potential
    imp_occp0 = np.zeros(nimp)       # non-interacting impurity occupation
    imp_occp1 = np.zeros(nimp)       # interacting impurity occupation
    # imp_entropy = 0.                 # impurity entropy
    H = np.zeros((n,n))
    V0 = np.zeros((n,n))             # non-interacting e-e interaction.
    if isinstance(U, np.ndarray):    # interacting e-e interaction.
        V1 = V0.copy()              
        V1[:nimp,:nimp] = U
    if isinstance(U, dict):
        V1 = defaultdict(lambda : np.zeros((n,n)))
        for k in U.keys():
            V1[k][:nimp,:nimp] += U[k]
    nbath = (n-nimp)//nimp          # # of bath sites per impurity.
    subsystem_pos = np.zeros((nimp,nbath+1),int) # site indices fo each subsytem.
    for i in range(nimp):
        bl = i*nbath+nimp
        bh = bl + nbath
        pos = np.arange(bl,bh)
        subsystem_pos[i,:] = np.insert(pos, 0, i)
    gf0 = []
    gfimp = []
    for i in range(nimp):
        rs = RandomSampler(dos[i], [z.real[0],z.real[-1]], nbath+1, rng) 
        gf0.append(Gf0(rs))
        gfimp.append(Gfimp(nbath+1))
    neig1 = None #np.ones((n+1)*(n+1),int) * 3
    neig0 = None #np.ones((n+1)*(n+1),int) * 3
    for _ in range(N):
        found = False
        while not found:
            for i in range(nimp):
                gf0[i].sample()
                gfimp[i].fit(gf0[i])
            build_moam(H, gfimp)
            espace, egs = build_espace(H, V0, neig0)
            # screen_espace(espace, egs, beta)
            # adjust_neigsector(espace, neig0, n)
            _, sct = next((k,v) for k,v in espace.items() if abs(v.eigvals[0]-egs)<1e-7)
            evec = sct.eigvecs[:,0]
            occps = [[get_occupation(evec,sct.states.up,sct.states.dw,i) for i in pos]
                     for pos in subsystem_pos]
            N0 = [sum(subsystem_occps) for subsystem_occps in occps]
            occp0 = [subsystem_occps[0] for subsystem_occps in occps]
            H.flat[:nimp*(n+1):n+1] -= sigma0
            espace, egs = build_espace(H, V1, neig1)
            # adjust_neigsector(espace, neig1, n)
            _, sct = next((k,v) for k,v in espace.items() if abs(v.eigvals[0]-egs)<1e-7)
            evec = sct.eigvecs[:,0]
            occps = [[get_occupation(evec,sct.states.up,sct.states.dw,i) for i in pos]
                     for pos in subsystem_pos]
            N1 = [sum(subsystem_occps) for subsystem_occps in occps]
            occp1 = [subsystem_occps[0] for subsystem_occps in occps]
            screen_espace(espace, egs, beta)
            for i in range(nimp):
                if np.allclose(N1[i],N0[i]):
                    gf = build_gf_lanczos(H,V1,espace,beta,egs,pos=i)
                    sigma[i] += np.reciprocal(gf0[i](z))-np.reciprocal(gf(z.real,z.imag))
                    imp_occp0[i] += occp0[i]
                    imp_occp1[i] += occp1[i]
                    # imp_entropy += get_entropy(gfimp,espace,egs,beta)
                    found = True
    sigma /= N
    imp_occp0 /= N
    imp_occp1 /= N
    # imp_entropy /= (N*kB)
    imp_entropy = 0. #(N*kB)
    if return_sigma:
            return sigma, imp_occp0, imp_occp1, imp_entropy
    return imp_occp0, imp_occp1, imp_entropy


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
