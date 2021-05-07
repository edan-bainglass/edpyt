from edpyt.gf_lanczos import build_gf_lanczos
import numpy as np
from scipy.optimize import root_scalar, broyden1

from edpyt.integrate_gf import integrate_gf
from edpyt.fit import fit_hybrid, fit_weiss_inv, g0inv_discrete
from edpyt.espace import adjust_neigsector, build_espace, screen_espace


def adjust_mu(gf, occupancy_goal, bracket=(-20.,20)):
    """Get the chemical potential to obtain the occupancy goal.
    
    NOTE : The gf is supposed to have the general form
    (z+mu-Delta(z)-Sigma(z))^-1. Here, `distance` returns
    the change in mu required to satisfy the occupancy goal.
    """
    distance = lambda mu: np.sum((2 * integrate_gf(gf, mu))-occupancy_goal)
    return root_scalar(distance, bracket=bracket, method='brentq').root + gf.mu


<<<<<<< HEAD
def break_spin_symmetry(H, sign, field):
    """Break spin symmetry by applying a symmetry breaking field.
    
    ack.: https://github.com/QcmPlab/LIB_DMFT_ED/blob/master/src/ED_BATH/user_aux.f90
    """
    assert H.ndim==3, "Hamiltonina must have spin index along 1st dimension."
    n = H.shape[-1]
    H[0,(n+1)::(n+1)] += sign * field
    H[1,(n+1)::(n+1)] -= sign * field

=======
def break_SU2_symmetry(H):
    assert H.ndim==3, "Hamiltonina mush "
>>>>>>> 196babc... dev: add ispin build mb hamilton

class Converged(Exception):
  def __init__(self, message):
      self.message = message


class FailedToConverge(Exception):
  def __init__(self, message):
      self.message = message


class Gfimp:
    """Green's function of SIAM model.

    """
    def __init__(self, n, nmats=3000, U=3., beta=1e6, neig=None, nspin=1):
        self.n = n
        self.nmats = nmats # Used in Matsubara fit. 
        self.beta = beta # Used in Matsubara fit and interacting green's function.
        self.H = np.zeros((n,n))
        self.V = np.zeros((n,n))
        self.V[0,0] = U
        self.Delta = None
        self.neig = neig # used in diagonalization

    def fit(self, Delta):
        """Fit hybridization and update bath params."""
        #                __ n-1           2  
        #           !    \               Vk  
        # Delta(z)  =                 -------
        #                /__k = 0      z - ek
        n = self.n
        Delta_disc = fit_hybrid(Delta, n-1, self.nmats, self.beta)
        self.Delta = Delta_disc
        self.H[1:,0] = self.H[0,1:] = self.vk
        self.H.flat[(n+1)::(n+1)] = self.ek

    def fit_weiss_inv(self, Weiss):
        """Fit hybridization and update bath params."""
        # g0inv = (z-e0-Delta(z))
        n = self.n
        g0inv_disc = fit_weiss_inv(Weiss, n-1, self.nmats, self.beta)
        self.update(-g0inv_disc.a0)
        self.Delta = g0inv_disc.Delta
        self.H[1:,0] = self.H[0,1:] = self.vk
        self.H.flat[(n+1)::(n+1)] = self.ek

    @property
    def vk(self):
        return self.Delta.b

    @property
    def ek(self):
        return self.Delta.a

    @property
    def mu(self):
        # ed = -mu
        return -self.H[0,0]

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
        gf = build_gf_lanczos(H, V, espace, self.beta, egs, repr='sp')
        self.Sigma = lambda z: np.reciprocal(self.free(z))-np.reciprocal(gf(z.real,z.imag))


class Gfloc:
    """Parent local green's function.
    """
    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to local self-energy!"""
        self.Sigma = Sigma

    def Delta(self, z):
        """Hybridization."""
        #                               -1
        # Delta(z) = z+mu - Sigma(z) - g (z)
        #
        return z+self.mu-self.Weiss(z)

    def Weiss(self, z):
        """Weiss filed."""
        gloc_inv = self(z, inverse=True)
        return self.Sigma(z)+gloc_inv
    
    
# Analytical Bethe lattice
_ht = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2))
eps = 1e-20
ht = lambda z: _ht(z.real+1.j*(z.imag if z.imag>0. else eps))


class Gfhybrid(Gfloc):
    """Local green's function defined by hybridization.
    """
    #          __     
    #         |                  1
    # G(z) =  |   de  -----------------------------
    #       __|       z + mu - Simga(z) - Hybrid(z) 
    # 
    def __init__(self, Hybrid):
        self.Hybrid = Hybrid

    def __call__(self, z, inverse=False):
        """Interacting green's function."""
        gloc_inv = self.free(z,inverse=True)-self.Sigma(z)
        if inverse:
            return gloc_inv
        return np.reciprocal(gloc_inv)

    def free(self, z, inverse=False):
        g0inv = z+self.mu-self.Hybrid(z)
        if inverse:
            return g0inv
        return np.reciprocal(g0inv)


class Gfhilbert(Gfloc):
    """Local green's function defined by hilbert transform.
    """
    #          __     
    #         |            dos(e)
    # G(z) =  |   de  ----------------
    #       __|       z + mu - Simga(z) 
    # 
    def __init__(self, hilbert):
        self.hilbert = hilbert

    def __call__(self, z, inverse=False):
        """Interacting green's function."""
        gloc = self.hilbert(z+self.mu-self.Sigma(z))
        if inverse:
            return np.reciprocal(gloc)
        return gloc

    def free(self, z, inverse=False):
        g0 = self.hilbert(z+self.mu)
        if inverse:
            return np.reciprocal(g0)
        return g0


def dmft_step(delta, gfimp, gfloc, occupancy_goal=None):
    """Perform a DMFT self-consistency step.
    
    Fit the hybridization function with a discrete bath and
    solve the impurity associated problem. If a target
    occupation is given, adjust mu of both local and impurity.
    """
    gfimp.fit(delta) # at matsubara frequencies
    gfimp.solve()
    gfloc.set_local(gfimp.Sigma)
    if occupancy_goal is not None:
        mu = adjust_mu(gfimp, occupancy_goal)
        gfimp.update(mu)
        gfloc.update(mu)


class DMFT:
    
    def __init__(self, gfimp, gfloc, occupancy_goal=None, max_iter=20, tol=1e-3):
        self.gfimp = gfimp
        self.gfloc = gfloc
        self.occupancy_goal = occupancy_goal
        self.it = 0
        self.tol = tol
        self.max_iter = max_iter
        wn = (2*np.arange(gfimp.nmats)+1)*np.pi/gfimp.beta
        self.z = 1.j*wn

    def dmft_step(self, delta):
        print(f'Iteration : {self.it:2}')
        dmft_step(delta, self.gfimp, self.gfloc, self.occupancy_goal)
        delta_new = self.gfloc.Delta(self.z)
        if self.occupancy_goal is not None:
            occp = 2.*integrate_gf(self.gfimp)[0]
            print(f'Occupation : {occp:.5f} Chemical potential : {self.gfimp.mu:.5f}', end=' ')
        return delta_new

    def initialize(self, U):
        Sigma = lambda z: U * self.occupancy_goal / 2.
        mu = U/2.
        self.gfloc.set_local(Sigma)
        self.gfloc.update(mu)
        self.gfimp.update(mu)
        return self.gfloc.Delta(self.z)

    def __call__(self, delta):
        delta_new = self.dmft_step(delta)
        eps = np.linalg.norm(delta_new - delta)
        print(f'Error : {eps:.5f}')
        if eps < self.tol:
            raise Converged('Converged!')
        self.it += 1
        if self.it > self.max_iter:
            raise FailedToConverge('Failed to converge!')
        return delta_new

    def solve_with_broyden_mixing(self, delta, alpha=0.5, verbose=True, callback=None):
        distance = lambda delta: self(delta)-delta
        broyden1(distance, delta, alpha=alpha, reduction_method="svd", 
                max_rank=10, verbose=verbose, f_tol=1e-99, callback=callback) # Loop forever (small f_tol!)

    def solve_with_linear_mixing(self, delta, alpha=0.5):
        delta_in = delta
        while True:
            delta_out = self(delta)
            delta_in = alpha * delta_out + (1-alpha) * delta_in

    def solve(self, delta, mixing_method='broyden', **kwargs):
        """'linear' or 'broyden' mixing 
        the quantity being mixed is the hybridisation function
        
        """
        if mixing_method == 'linear':
            self.solve_with_linear_mixing(delta, **kwargs)
        elif mixing_method == 'broyden':
            self.solve_with_broyden_mixing(delta, **kwargs)

