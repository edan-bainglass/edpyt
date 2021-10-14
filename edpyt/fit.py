#!/usr/bin/env python
from numba.np.ufunc import parallel
import numpy as np
from math import sqrt

from numba import njit, prange
from numba.types import complex128, float64, Array
from numba.experimental import jitclass
from collections import namedtuple

from edpyt.fmin_bfgs import _minimize_bfgs

@njit#([(Array(float64, 1, 'C'), Array(complex128, 1, 'A'), Array(complex128, 1, 'A'))],parallel=False)
def hybrid_discrete(p, z, out):
    #                ____ N=nbath           2
    #               \               |p(i+N)|
    #  Delta(z) =    \             ---------
    #                /              z - p(i)
    #               /____ i=0
    n = p.size//2
    for i in prange(z.size):
        out[i] = 0.+0.j
        for j in range(n):
            out[i] += p[j+n]**2/(z[i]-p[j])


class Delta:
    """Discrete hybridization function.
    
    Args:
        p : array of poles (1st half, p[:N], N=# of bath sites) 
            and amplitudes (2nd half, p[N:], N=# of bath sites).
    """
    def __init__(self, p):
        self.p = p
        self.nbath = p.size//2
        
    @property
    def ek(self):
        return self.p[:self.nbath]
    
    @property
    def vk(self):
        return self.p[self.nbath:]

    def __call__(self, z):
        out = np.empty_like(z)
        hybrid_discrete(self.p, z, out)
        return out
    

spec = [
    ('z',complex128[:]),
    ('out',complex128[:]),
    ('vals_true',complex128[:]),
    ('weights',float64[:]),
]


@njit#([(Array(complex128, 1, 'A'), Array(complex128, 1, 'A'), Array(float64, 1, 'A'))],parallel=False)
def cityblock(a, b, w):
    out = 0.
    for i in prange(a.size):
        c = a[i] - b[i]
        out += w[i] * (c.real**2 + c.imag**2)
    return sqrt(out)


@jitclass(spec)
class Chi:
    """Cost function.
    
    Args:
        z : Matzubara frequencies.
        vals_true : True function values.
    """
    #                ____
    #               \
    #       chi =    \      | Delta   (z) - Delta  (z) |
    #                /            true          fit
    #               /____ z
    def __init__(self, z, vals_true, weights):
        self.z = z
        self.out = np.empty_like(z)
        self.vals_true = vals_true
        self.weights = weights
    
    def call(self, p):
        hybrid_discrete(p, self.z, self.out)
        return cityblock(self.out, self.vals_true, self.weights)
    
    def __call__(self, p):
        return self.call(p)


chi_type = Chi.class_type.instance_type


@njit(parallel=True)
def _fit_hybrid(vals_true, z, popt, weights):
    """Parallel fit.
    
    Args:
        vals_true : (array, shape=(nfit, nmats))
        z : (array, shape=(nmats))
        popt : (array, shape=(nfit, 2*nbath))
        
    """
    fopt = np.empty(vals_true.shape[0])
    for i in prange(vals_true.shape[0]):
    # for i in range(vals_true.shape[0]):
        chi = Chi(z, vals_true[i], weights)
        popt[i], fopt[i] = _minimize_bfgs(chi, popt[i])
        # output = fmin_bfgs(chi, popt[i], full_output=True)
        # popt[i], fopt[i] = output[0], output[1]
    return fopt


@njit('(float64[:],float64)', cache=True)
def set_initial_bath(p, bandwidth=2.):
    """
    Args:
        nbath : # of bath sites.
        bandwidth : half-bandwidth for the bath initialization.
    """
    nbath = p.size//2
    ek = p[:nbath]
    vk = p[nbath:]
    # Hoppings
    vk[:] = max(0.1, 1/np.sqrt(nbath))
    # Energies
    # ODD  : [-2,-1,0,+1,+2]
    # EVEN : [-2,-1,-0.1,0.1,+1,+2]
    ek[0] = -bandwidth
    ek[-1] = bandwidth
    nhalf = nbath//2
    nbath_is_odd = bool(nbath&1)
    nbath_is_even = not nbath_is_odd
    if nbath_is_even and nbath>=4:
        de = bandwidth/max(nhalf-1,1)
        ek[nhalf-1] = -0.1    
        ek[nhalf] = 0.1    
        for i in range(1,nhalf-1):
            ek[i] = -bandwidth + i*de
            ek[nbath-i-1] = bandwidth - i*de
    if nbath_is_odd and nbath>=3:
        de = bandwidth/nhalf
        ek[nhalf] = 0.
        for i in range(1,nhalf):
            ek[i] = -bandwidth + i*de
            ek[nbath-i-1] = bandwidth - i*de
    
    
def fit_hybrid(vals_true, nbath=7, nmats=3000, beta=70., tol_fit=5., max_fit=5, alpha=0., bandwidth=2.):
    """Fit hybridization using matsubara frequencies.

    Args:
        vals_true : True function values.
        nbath : # of bath sites.
        nmats : # of matsubara frequencies.
        beta : 1/kbT
        tol_fit : tollerance for fit error.
        max_fit : repeat fit optimization if error is greater than `tol_fit`
                  for at max. `max_fit` times. Then return anyway.
        alpha : weight factor for matzubara frequency, i.e. w^-alpha * |X(z) - Y(z)|.
    """
    squeeze_output = False
    if vals_true.ndim<2:
        squeeze_output = True
    vals_true = np.atleast_2d(vals_true)
    wn = (2*np.arange(nmats)+1)*(np.pi/beta)
    z = 1.j*wn
    weights = wn**-alpha
    shape = vals_true.shape[:-1]
    nfit = np.prod(shape)
    nparams = 2*nbath
    # generate_random = lambda n: (2*np.random.random(n*nparams)-1).reshape(n,nparams)
    vals_true = vals_true.reshape(nfit, nmats)
    popt = np.empty((nfit, nparams))
    fopt = np.ones(nfit) + tol_fit
    it = 0
    # while np.any(fopt>tol_fit)&(it<max_fit):
    need_fit = np.where(fopt>tol_fit)[0]
    # p = generate_random(need_fit.size)
    set_initial_bath(popt[0], bandwidth)
    for i in range(1,nfit):
        popt[i] = popt[0]
    fopt[need_fit] = _fit_hybrid(vals_true[need_fit], z, popt, weights)
    # popt[need_fit] = p[:]
    it += 1
    fopt.shape = shape    
    popt.shape = shape + (nparams,)
    if squeeze_output:
        return popt[0], fopt[0]
    return popt, fopt
    
    
# @njit(parallel=True)
# def run(z, vals_true, ntimes=20):
#     for i in prange(ntimes):
#         p0 = 2.*np.random.random(12)-1.
#         chi = Chi(z, vals_true)
#         _minimize_bfgs(chi, p0)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from edpyt.fit_serial import fit_hybrid as fit_hybrid_serial
    beta = 70.
    nmats = 3000
    hybrid_true = lambda z: 2*(z-np.sqrt(z**2-1))
    z = 1.j*(2*np.arange(nmats)+1)*np.pi/beta
    vals_true = hybrid_true(z)
    vals_true = np.tile(vals_true, (2,1))
    popt, fopt = fit_hybrid(vals_true, max_fit=1)
    vals_serial = fit_hybrid_serial(vals_true, nmats=nmats, beta=beta, 
                                    full_output=True)[0](z)
    vals_trial = Delta(popt[0])(z)
    fig, axs = plt.subplots(1,2)
    axs[0].plot(z.imag, vals_true[0].imag, z.imag, vals_trial.imag, z.imag, vals_serial.imag)
    axs[1].plot(z.imag, vals_true[0].real, z.imag, vals_trial.real, z.imag, vals_serial.real)
    plt.savefig('fit_hybrid.png')
    plt.close()
    print(fopt)




