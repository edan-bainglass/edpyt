import numpy as np
from scipy.optimize import fmin_bfgs
from functools import singledispatch
import typing

"""Look at DCore github for example codes."""


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


def g0inv_discrete(params):
    """Discrete impurity green's function.

    """
    #          
    #                  1
    #  g(z) =  ------------------
    #   0       z - a0 - Delta(z)
    #          
    a0 = params[0]
    params = params[1:]
    Delta = hybrid_discrete(params)
    def inner(z):
        return z-a0-Delta(z)
    inner.a0 = a0
    inner.Delta = Delta
    return inner


def cost_chi(func_discrete, vals_true, z):
    """Distance between true and fitted hybridization.

    """
    #                ____
    #               \
    #       chi =    \      | Delta   (z) - Delta  (z) |
    #                /            true          fit
    #               /____ z
    def inner(params):
        inner = func_discrete(params)
        return np.linalg.norm(inner(z) - vals_true)
    return inner


def _fit(vals_true: np.ndarray, func_discrete, z, nparams=7, **kwargs):
    # initial guess, random values in the range [-1:1]
    p0 = 2*np.random.random(nparams)-1
    chi = cost_chi(func_discrete, vals_true, z)
    output = fmin_bfgs(chi, p0, **kwargs)
    if kwargs.get('full_output', False):
        return func_discrete(output[0]), output[1] 
    return output

def fit(x_true, func_discrete, nparams, nmats=100, beta=10, **kwargs):
    """Fit function using matsubara frequencies.

    Args:
        x_true : (callable or np.ndarray)
            function or values to fit with matsubara frequencies.
        func_discrete : (callable)
            function factory used to fit x_true. At each call, 
            the function must accept a set of input parameters (and 
            save them locally) and returns a callable function that
            performs the mapping f(x)=y.
        nparams : # of parameters for discrete function.
        nmats : # of matsubara frequencies to use for fitting.
        beta : 1/kbT
        kwargs : extra parameters to pass to the optimizer.

    Return:
        func_discrete : discrete function with optimized local
            parameters.

    """
    wn = (2*np.arange(nmats)+1)*np.pi/beta
    z = 1.j*wn
    if isinstance(x_true, typing.Callable):
        vals_true = x_true(z)
    elif isinstance(x_true,np.ndarray):
        vals_true = x_true
    else:
        raise ValueError('Invalid input function for fit.')
    return _fit(vals_true, func_discrete, z, nparams, **kwargs)    


def fit_hybrid(hybrid_true,  nbath=7, nmats=100, beta=10, **kwargs):
    """Fit hybridization bath using matsubara frequencies."""
    nparams = 2*nbath
    return fit(hybrid_true, hybrid_discrete, nparams, nmats, beta, **kwargs)


def fit_weiss_inv(weiss_inv_true,  nbath=7, nmats=100, beta=10, **kwargs):
    """Fit inverse weiss filed using matsubara frequencies."""
    nparams = 2*nbath+1
    return fit(weiss_inv_true, g0inv_discrete, nparams, nmats, beta, **kwargs)