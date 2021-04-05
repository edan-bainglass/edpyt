from scipy.linalg.special_matrices import toeplitz
from edpyt.gf_lanczos import build_gf_lanczos
import numpy as np
from scipy.optimize import root_scalar, broyden1

from edpyt.integrate_gf import integrate_gf
from edpyt.fit import fit_hybrid
from edpyt.espace import adjust_neigsector, build_espace, screen_espace


class Gfimp:
    """Green's function of SIAM model.

    """
    def __init__(self, n, nmats=100, U=3., beta=1e6, neig=None):
        self.n = n
        self.nmats = nmats # sed in Matsubara fit. 
        self.beta = beta # sed in Matsubara fit and interacting green's function.
        self.H = np.zeros((n,n))
        self.V = np.zeros((n,n))
        self.V[0,0] = U
        self.Delta = None
        self.neig = neig # used in diagonalization

    def fit(self, Delta):
        """Fit hybridization and update bath params.
        """
        #                __ n-1           2  
        #           !    \               Vk  
        # Delta(z)  =                 -------
        #                /__k = 0      z - ek
        n = self.n
        Delta_disc = fit_hybrid(Delta, n-1, self.nmats, self.beta)
        self.Delta = Delta_disc
        self.H[1:,0] = self.H[0,1:] = self.vk
        self.H.flat[(n+1)::(n+1)] = self.ek

    @property
    def vk(self):
        return self.Delta.b

    @property
    def ek(self):
        return self.Delta.a

    def update(self, mu):
        """Update chemical potential."""
        self.H[0,0] = -mu

    def fit_update(self, Delta, mu):
        """Update impurity model."""
        self.fit(Delta)
        self.update(mu)

    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #                 1
        # g  =    -----------------
        #         z + mu - Delta(z)
        g0_inv = z-self.H[0,0]-self.Delta(z)
        if inverse:
            return g0_inv
        return np.reciprocal(g0_inv)

    def __call__(self, z):
        """Interacting green's function."""
        #                    1
        # g  =    ----------------------------
        #  0      z + mu - Delta(z) - Sigma(z)
        return np.reciprocal(z-self.H[0,0]-self.Delta(z)-self.Sigma(z))

    def solve(self):
        """Solve impurity model.
        
        """
        #                  -1            -1
        #  Sigma(z)    =  g (z)     -   g (z)
        #                  0 
        H, V = self.H, self.V
        espace, egs = build_espace(H, V, self.neig)
        screen_espace(espace, egs)
        adjust_neigsector(espace, self.neig, self.n)
        gf = build_gf_lanczos(H, V, espace, self.beta, egs)
        self.Sigma = lambda z: np.reciprocal(self.free(z))-np.reciprocal(gf(z.real,z.imag))


class Gfhilbert:
    """Local green's function defined by hilbert transform.
    """
    #          __     
    #         |            dos(e)
    # G(z) =  |   de  ----------------
    #       __|       z + mu - Simga(z) 
    # 
    def __init__(self, hilbert):
        self.hilbert = hilbert

    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to local self-energy!"""
        self.Sigma = Sigma

    def __call__(self, z):
        """Interacting green's function."""
        return self.hilbert(z+self.mu-self.Sigma(z))

    def Delta(self, z):
        """Hybridization."""
        #             -1                 -1
        # Delta(z) = g (z) - Sigma(z) - g (z)
        #             0
        gloc_inv = np.reciprocal(self(z))
        return z+self.mu-self.Sigma(z)-gloc_inv

    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #               1
        #  g (z) = ------------------
        #   0       z + mu - Delta(z)
        g0_inv = z+self.mu-self.Delta(z)
        if inverse:
            return g0_inv
        return np.reciprocal(g0_inv)


def adjust_mu(gf, occupancy_goal, mu=0.):
    """Get the chemical potential to obtain the occupancy goal.
    
    """
    distance = lambda mu: (2 * integrate_gf(gf, mu))-occupancy_goal
    return root_scalar(distance, x0=mu, x1=mu-0.1).root
    
    
# Analytical Bethe lattice
_ht = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2))
eps = 1e-20
ht = lambda z: _ht(z.real+1.j*(z.imag if z.imag>0. else eps))


class Converged(Exception):
  def __init__(self, message):
      self.message = message


class FailedToConverge(Exception):
  def __init__(self, message):
      self.message = message


# def dmft_step(z, sigma, gfimp, gfloc, occupancy_goal):
#     """Perform a DMFT self-consistency step."""
#     gfloc.set_local(sigma)
#     mu = adjust_mu(gfloc, occupancy_goal)
#     gfloc.update(mu)
#     Delta = gfloc(z)
#     gfimp.fit_update(Delta, mu)
#     gfimp.solve()


def dmft_step(delta, gfimp, gfloc, occupancy_goal):
    """Perform a DMFT self-consistency step."""
    gfimp.fit(delta) # at matsubara frequencies
    gfimp.solve()
    mu = adjust_mu(gfimp, occupancy_goal)
    gfimp.update(mu)
    gfloc.set_local(gfimp.Sigma)
    gfloc.update(mu)


class DMFT:
    
    def __init__(self, gfimp, gfloc, occupancy_goal, max_iter=20, tol=1e-3):
        self.gfimp = gfimp
        self.gfloc = gfloc
        self.occupancy_goal = occupancy_goal
        self.it = 0
        self.tol = tol
        self.max_iter = max_iter
        wn = (2*np.arange(gfimp.nmats)+1)*np.pi/gfimp.beta
        self.z = 1.j*wn

    # def __call__(self, sigma):
    #     dmft_step(self.z, sigma, self.gfimp, self.gfloc, self.occupancy_goal)
    #     sigma_new = self.gfimp.Sigma(self.z)
    #     eps = np.linalg.norm(sigma_new - sigma)
    #     if eps < self.tol:
    #         raise Converged('converged!')
    #     if self.it > self.max_iter:
    #         raise FailedToConverge('failed!')
    #     return sigma_new

    # def solve(self, sigma, alpha=0.5, verbose=True):
    #     distance = lambda sigma: self(sigma)-sigma
    #     broyden1(distance, sigma, alpha=alpha, reduction_method="svd", 
    #             max_rank=10, verbose=verbose, f_tol=1e-99) # Loop forever (small f_tol!)

    def initialize(self):
        U = self.gfimp.V[0,0]
        Sigma = lambda z: U * self.occupancy_goal / 2.
        mu = U/2.
        self.gfloc.set_local(Sigma)
        self.gfloc.update(mu)
        self.gfimp.update(mu)
        return self.gfloc.Delta(self.z)

    def __call__(self, delta):
        dmft_step(delta, self.gfimp, self.gfloc, self.occupancy_goal)
        delta_new = self.gfloc.Delta(self.z)
        eps = np.linalg.norm(delta_new - delta)
        if eps < self.tol:
            raise Converged('converged!')
        if self.it > self.max_iter:
            raise FailedToConverge('failed!')
        return delta_new

    def solve(self, sigma, alpha=0.5, verbose=True):
        distance = lambda sigma: self(sigma)-sigma
        broyden1(distance, sigma, alpha=alpha, reduction_method="svd", 
                max_rank=10, verbose=verbose, f_tol=1e-99) # Loop forever (small f_tol!)

