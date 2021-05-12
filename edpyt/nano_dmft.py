import numpy as np

from edpyt.integrate_gf import integrate_gf
from edpyt.dmft import _DMFT, adjust_mu

from functools import wraps


def vectorize(signature=None, kwarg=None):
    """Vectorize output of a function.
    
    If `kwarg` is not None, check if `kwarg` is absent when 
    target function is called (wrapper) and automatically add 
    default `kwarg` to match signature's # of input arguments.

    NOTE : `signature` must include the keyword 
    argument, if this is present.

    Args:
        signature : (str)
            Generalized universal function signature.
        kwargs : (dict, optional)
            Default keyword argument.
    
    """
    def decorator(fn):
        vectorized = np.vectorize(fn, signature=signature)
        if kwarg is not None:
            @wraps(fn)
            def wrapper(*args,**kw):
                # Add default `kwarg` if kw is not specified.
                _kw = kw or kwarg
                return vectorized(*args,**_kw)
        else:
            @wraps(fn)
            def wrapper(*args):
                return vectorized(*args)
        return wrapper
    return decorator


class Gfloc:
    """nano Local lattice green's function.

        H : Hamiltonian matrix
        S : overlap matrix
        Hybrid : (callable) hybridization funciton, must
            return a matrix of the same dimensions of H(S).
        (below, see also np.unique)
        idx_neq : the indices of the input array that give the unique values
        idx_inv : the indices of the unique array that reconstruct the input array
    """
    def __init__(self, H, S, Hybrid, idx_neq, idx_inv) -> None:
        self.n = H.shape[-1]
        self.H = H
        self.S = S
        self.Hybrid = Hybrid
        self.idx_neq = idx_neq
        self.idx_inv = idx_inv

    @property
    def ed(self):
        return self.H.diagonal()[self.idx_neq]

    # @vectorize(signature='(),(),()->(n,n)',kwarg=dict(inverse=False))
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

    @vectorize(signature='(),()->(n)')
    def Delta(self, z):
        """Hybridization."""
        #                                       -1
        # Delta(z) = z+mu - Sigma(z) - ( G (z) )
        #                                 ii
        # gloc_inv = np.reciprocal(self(z).diagonal())[self.idx_neq]
        return z+self.mu-self.ed-self.Weiss(z)

    @vectorize(signature='(),()->(n)')
    def Weiss(self, z):
        """Weiss field."""
        #  -1                             -1
        # G     (z) = Sigma(z) + ( G (z) )
        #  0,ii                     ii
        gloc_inv = np.reciprocal(self(z).diagonal())[self.idx_neq]
        return gloc_inv+self.Sigma(z)

    # @vectorize(signature='(),(),()->(n,n)',kwarg=dict(inverse=False))
    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #                                       -1
        #  g (z) = ((z + mu)*S - H - Hybrid(z))
        #   0      
        g0_inv = (z+self.mu)*self.S-self.H-self.Hybrid(z)
        if inverse:
            return g0_inv
        return np.linalg.inv(g0_inv)

    # Helper

    def integrate(self, mu=0.):
        return 2. * integrate_gf(self, mu)


class Gfimp:

    def __init__(self, gfimp) -> None:
        self.gfimp = gfimp
        # self.Delta = np.vectorize(self._Delta, signature='()->(n)')
        # self.Sigma = np.vectorize(self._Sigma, signature='()->(n)')

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
            gf.fit(delta[...,i])

    # def fit_weiss_inv(self, weiss_inv):
    #     """Fit discrete bath."""
    #     for i, gf in enumerate(self):
    #         gf.fit_weiss_inv(np.ascontiguousarray(weiss_inv[:,i]))

    # @vectorize(signature='(),()->(n)')
    def Delta(self, z):
        """Hybridization."""
        return np.fromiter((gf.Delta(z) for gf in self.gfimp), complex)

    # @vectorize(signature='(),()->(n)')
    def Sigma(self, z):
        """Correlated self-energy."""
        return np.fromiter((gf.Sigma(z) for gf in self.gfimp), complex)

    # @vectorize(signature='(),(),()->(n)',kwarg=dict(inverse=False))
    # def free(self, z, inverse=False):
    #     """Correlated self-energy."""
    #     return np.fromiter((gf.free(z, inverse=inverse) for gf in self.gfimp), complex)

    def __getitem__(self, i):
        return self.gfimp[i]

    def __len__(self):
        return len(self.gfimp)

    def __iter__(self):
        yield from iter(self.gfimp)


def dmft_step(delta, gfimp, gfloc):
    """Perform a DMFT self-consistency step."""
    for i, gf in enumerate(gfimp):
        gf.fit(delta[:,i])
        gf.update(gfloc.mu-gfloc.ed[i])
        gf.solve()
    gfloc.set_local(gfimp.Sigma)


def dmft_step_adjust(delta, gfimp, gfloc, occupancy_goal):
    """Perform a DMFT self-consistency step and adjust chemical potential to 
    target occupation (for both local and impurity green's functions."""
    dmft_step(delta, gfimp, gfloc)
    mu = adjust_mu(gfloc, occupancy_goal)
    gfloc.update(mu)


def dmft_step_adjust_ext(delta, gfimp, gfloc):
    """Set chemical potential and perform a DMFT self-consistency step.
    The chemical potential is the first entry of the delta array and is
    adjusted by an external minimizer during the self consistency loop."""
    mu, delta = delta[0], delta[1:].reshape(-1,len(gfimp))
    gfimp.update(mu)
    gfloc.update(mu)
    dmft_step(delta, gfimp, gfloc)


class _nanoDMFT(_DMFT):

    def step(self, delta):
        dmft_step(delta, self.gfimp, self.gfloc)
        delta_new = self.gfloc.Delta(self.z)
        occp = sum(self.gfloc.integrate())
        return occp, delta_new


class _nanoDMFTadjust(_DMFT):

    def step(self, delta):
        dmft_step_adjust(delta, self.gfimp, self.gfloc, self.occupancy_goal[self.gfloc.idx_inv])
        delta_new = self.gfloc.Delta(self.z)
        occp = sum(self.gfloc.integrate())
        return occp, delta_new


class _nanoDMFTadjustext(_DMFT):

    def initialize(self, U):
        delta = np.empty(1+self.z.size*len(self.gfimp), complex)
        delta[1:] = super().initialize(U).flat
        delta[0] = self.gfimp.mu
        return delta

    def step(self, delta):
        dmft_step_adjust_ext(delta, self.gfimp, self.gfloc)
        delta_new = np.empty_like(delta)
        delta_new[1:] = self.gfloc.Delta(self.z)
        delta_new[0] = sum(self.gfimp.integrate())
        return delta_new[0], delta_new

    def distance(self, delta):
        eps = self(delta) - delta
        eps[0] = sum(delta[0] - self.occupancy_goal)
        return eps


def solve(gfimp, gfloc, occupancy_goal=None, max_iter=20, tol=1e-3, 
          occp_method=None, mixing_method='broyden'):
    
    if occp_method is None:
        dmft_solver = _nanoDMFT(gfimp, gfloc, max_iter=max_iter, tol=tol)
    
    elif occp_method == 'adjust':
        dmft_solver = _nanoDMFTadjust(gfimp, gfloc, occupancy_goal, max_iter, tol)    
    
    elif occp_method == 'adjust_ext':
        dmft_solver = _nanoDMFTadjustext(gfimp, gfloc, occupancy_goal, max_iter, tol)
    
    else:
        raise ValueError("Invalid occupation method. Choose between None, adjust, adjust_ext.")

    delta = dmft_solver.initialize()
    dmft_solver.solve(delta, mixing_method=mixing_method)