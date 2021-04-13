from inspect import signature
import numpy as np
from numpy.lib.function_base import vectorize
from scipy.optimize import root_scalar

from edpyt.integrate_gf import integrate_gf
from edpyt.dmft import (
    Converged, DMFT, FailedToConverge, adjust_mu
)

from functools import wraps


def vectorize(otypes=None, signature=None):
    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature, excluded='self')
        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)
        return wrapper
    return decorator


class Gfloc:
    """Local green's function from hybridization matrix.
    """
    def __init__(self, H, S, Hybrid, idx_neq, idx_inv) -> None:
        self.n = H.shape[0]
        self.H = H
        self.S = S
        self.Hybrid = Hybrid
        self.idx_neq = idx_neq
        self.idx_inv = idx_inv

    @property
    def ed(self):
        return self.H.diagonal()[self.idx_neq]

    def __call__(self, z, inverse=False):
        """Interacting Green's function."""
        x = self.free(z, inverse=True)
        x.flat[::(self.n+1)] -= self.Sigma(z)[self.idx_inv]
        if inverse:
            return x
        return  np.linalg.inv(x)

    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to diagonal elements of local self-energy!"""
        self.Sigma = Sigma

    @vectorize(signature='(),()->(n)') # first argument is self
    def Delta(self, z):
        """Hybridization."""
        #                                       -1
        # Delta(z) = z+mu - Sigma(z) - ( G (z) )
        #                                 ii
        # gloc_inv = np.reciprocal(self(z).diagonal())[self.idx_neq]
        return z+self.mu-self.ed-self.Weiss(z)

    @vectorize(signature='(),()->(n)') # first argument is self
    def Weiss(self, z):
        """Weiss field."""
        #  -1                             -1
        # G     (z) = Sigma(z) + ( G (z) )
        #  0,ii                     ii
        gloc_inv = np.reciprocal(self(z).diagonal())[self.idx_neq]
        return gloc_inv+self.Sigma(z)

    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #                                       -1
        #  g (z) = ((z + mu)*S - H - Hybrid(z))
        #   0      
        g0_inv = (z+self.mu)*self.S-self.H-self.Hybrid(z)
        if inverse:
            return g0_inv
        return np.linalg.inv(g0_inv)


class Gfimp:

    def __init__(self, gfimp) -> None:
        self.gfimp = gfimp

    def __getattr__(self, name):
        """Default is to return attribute of first impurity."""
        return getattr(self.gfimp[0], name)

    def update(self, mu):
        """Updated chemical potential."""
        for gf in self:
            gf.update(mu)

    def fit(self, delta):
        """Fit discrete bath."""
        for i, gf in enumerate(self):
            gf.fit(np.ascontiguousarray(delta[:,i]))

    def fit_weiss_inv(self, weiss_inv):
        """Fit discrete bath."""
        for i, gf in enumerate(self):
            gf.fit_weiss_inv(np.ascontiguousarray(weiss_inv[:,i]))

    @vectorize(signature='(),()->(n)')
    def Delta(self, z):
        """Hybridization."""
        return np.fromiter((gf.Delta(z) for gf in self), complex)

    @vectorize(signature='(),()->(n)')
    def Sigma(self, z):
        """Correlated self-energy."""
        return np.fromiter((gf.Sigma(z) for gf in self), complex)

    @vectorize(signature='(),(),()->(n)')
    def free(self, z, inverse=False):
        """Correlated self-energy."""
        return np.fromiter((gf.free(z, inverse=inverse) for gf in self), complex)

    def __getitem__(self, i):
        return self.gfimp[i]

    def __iter__(self):
        yield from self.gfimp


def dmft_step(delta, gfimp, gfloc, occupancy_goal):
    """Perform a DMFT self-consistency step."""
    for i, gf in enumerate(gfimp):
        gf.fit(delta[:,i])
        gf.update(gfloc.mu-gfloc.ed[i])
        gf.solve()
    gfloc.set_local(gfimp.Sigma)


class nanoDMFT(DMFT):

    @staticmethod
    def dmft_step(*args, **kwargs):
        return dmft_step(*args, **kwargs)