#!/usr/bin/env python
from operator import ne
import numpy as np


from numba import njit, complex128, prange
from numba.experimental import jitclass
from collections import namedtuple

from edpyt.fmin_bfgs import _minimize_bfgs
from scipy.optimize import fmin_bfgs

@njit
def hybrid_discrete(p, z, out):
    #                ____ N=nbath           2
    #               \               |p(i+N)|
    #  Delta(z) =    \             ---------
    #                /              z - p(i)
    #               /____ i=0
    n = p.size//2
    for i in range(z.size):
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
]

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
    def __init__(self, z, vals_true):
        self.z = z
        self.out = np.empty_like(z)
        self.vals_true = vals_true
    
    def call(self, p):
        hybrid_discrete(p, self.z, self.out)
        return np.linalg.norm(self.out - self.vals_true)
    
    def __call__(self, p):
        return self.call(p)


@njit(parallel=True)
def _fit_hybrid(vals_true, z, popt):
    """Parallel fit.
    
    Args:
        vals_true : (array, shape=(nfit, nmats))
        z : (array, shape=(nmats))
        popt : (array, shape=(nfit, 2*nbath))
        
    """
    fopt = np.empty(vals_true.shape[0])
    for i in prange(vals_true.shape[0]):
    # for i in range(vals_true.shape[0]):
        chi = Chi(z, vals_true[i])
        popt[i], fopt[i] = _minimize_bfgs(chi, popt[i])
        # output = fmin_bfgs(chi, popt[i], full_output=True)
        # popt[i], fopt[i] = output[0], output[1]
    return fopt
    
 
def fit_hybrid(vals_true, nbath=7, nmats=3000, beta=70., tol_fit=5., max_fit=5):
    """Fit hybridization using matsubara frequencies.

    Args:
        vals_true : True function values.
        nbath : # of bath sites.
        nmats : # of matsubara frequencies.
        beta : 1/kbT
        tol_fit : tollerance for fit error.
        max_fit : repeat fit optimization if error is greater than `tol_fit`
                  for at max. `max_fit` times. Then return anyway.
    """
    squeeze_output = False
    if vals_true.ndim<2:
        squeeze_output = True
    vals_true = np.atleast_2d(vals_true)
    z = 1.j*(2*np.arange(nmats)+1)*np.pi/beta
    shape = vals_true.shape[:-1]
    nfit = np.prod(shape)
    nparams = 2*nbath
    generate_random = lambda n: (2*np.random.random(n*nparams)-1).reshape(n,nparams)
    vals_true.shape = (nfit, nmats)
    popt = np.empty((nfit, nparams))
    fopt = np.ones(nfit) + tol_fit
    it = 0
    while np.any(fopt>tol_fit)&(it<max_fit):
        need_fit = np.where(fopt>tol_fit)[0]
        p = generate_random(need_fit.size)
        fopt[need_fit] = _fit_hybrid(vals_true[need_fit], z, p)
        popt[need_fit] = p[:]
        it += 1
    vals_true.shape = shape + (nmats,)
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
    hybrid_true = lambda z: 2*(z-np.sqrt(z**2-1))
    z = 1.j*(2*np.arange(3000)+1)*np.pi/70.
    vals_true = hybrid_true(z)
    vals_true = np.tile(vals_true, (4,1))
    _fit_hybrid(vals_true, z, )
    run.parallel_diagnostics(level=4)




