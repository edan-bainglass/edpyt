import numpy as np
from scipy.optimize import fmin_bfgs
from functools import singledispatch
import typing

def hybrid_discrete(params):
    """Discrete bath to fit.

    """
    #                ____ nbath        2
    #               \            |b(i)|
    #  Delta(z) =    \          ---------
    #                /           z - a(i)
    #               /____ i=0
    a = params[:params.size//2]
    b = params[params.size//2:]
    def inner(z):
        z = np.atleast_1d(z)
        return np.einsum('i,i,ki->k',b,np.conj(b),np.reciprocal(z[:,None]-a[None,:]))
    inner.a = a
    inner.b = b
    return inner


def cost_chi(hybrid_true_z, z):
    """Distance between true and fitted hybridization.

    """
    #                ____
    #               \
    #       chi =    \      | Delta   (z) - Delta  (z) |
    #                /            true          fit
    #               /____ z
    def inner(params):
        hybrid = hybrid_discrete(params)
        return np.linalg.norm(hybrid(z) - hybrid_true_z)
    return inner


@singledispatch
def fit_hybrid(hybrid_true: typing.Callable, nbath=7, nmats=100, beta=10, **kwargs):
    """Fit hybridization bath using matsubara frequencies.

    Args:
        hybrid_true : (callable)
        nbath : # of bath sites to fit.
        nmats : # of matsubara frequencies to use for fitting.
        beta : 1/kbT
        kwargs : extra parameters to pass to the optimizer.

    Return:
        hybrid_disc : callable function.

        Holds discretization coefficients:
            hybrid_disc.a
            hybrid_disc.b

    """
    p0 = np.random.random(2*nbath)
    wn = (2*np.arange(nmats)+1)*np.pi/beta
    z = 1.j*wn
    hybrid_true_z = hybrid_true(z)
    chi = cost_chi(hybrid_true_z, z)
    popt = fmin_bfgs(chi, p0, **kwargs)
    return hybrid_discrete(popt)


@fit_hybrid.register
def _(hybrid_true: np.ndarray, nbath=7, nmats=100, beta=10, **kwargs):
    """Fit hybridization bath using matsubara frequencies.

    Args:
        hybrid_true : hybridization at Matsubara frequencies.
        nbath : # of bath sites to fit.
        nmats : (optional) used to check len(hybrid_true)
        beta : 1/kbT
        kwargs : extra parameters to pass to the optimizer.

    Return:
        hybrid_disc : callable function.

        Holds discretization coefficients:
            hybrid_disc.a
            hybrid_disc.b

    """
    assert len(hybrid_true) == nmats, "Lengths of Matsubara frequencies and hybridization values don't match"
    p0 = np.random.random(2*nbath)
    wn = (2*np.arange(nmats)+1)*np.pi/beta
    z = 1.j*wn
    chi = cost_chi(hybrid_true, z)
    popt = fmin_bfgs(chi, p0, **kwargs)
    return hybrid_discrete(popt)