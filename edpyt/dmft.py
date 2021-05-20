import numpy as np
from scipy.optimize import root_scalar, broyden1

from edpyt.integrate_gf import integrate_gf
from edpyt.fit import fit_hybrid
from edpyt.espace import adjust_neigsector, build_espace, screen_espace
from edpyt.gf_lanczos import build_gf_lanczos


def adjust_mu(gf, occupancy_goal, bracket=(-20.,20)):
    """Get the chemical potential to obtain the occupancy goal.
    
    NOTE : The gf is supposed to have the general form
    (z+mu-Delta(z)-Sigma(z))^-1. Here, `distance` returns
    the change in mu required to satisfy the occupancy goal.
    """
    distance = lambda mu: np.sum(gf.integrate(mu)-occupancy_goal)
    return root_scalar(distance, bracket=bracket, method='brentq').root + gf.mu


class Converged(Exception):
  def __init__(self, message):
      self.message = message


class FailedToConverge(Exception):
  def __init__(self, message):
      self.message = message


class Gfimp:
    """Green's function of SIAM model.

    """

    def __init__(self, n, nmats=3000, U=3., beta=1e6, neig=None, tol_fit=10., max_fit=3):
        self.n = n
        self.nmats = nmats # Used in Matsubara fit. 
        self.beta = beta # Used in Matsubara fit and interacting green's function.
        self.H = np.zeros((n,n))
        self.V = np.zeros((n,n))
        self.V[0,0] = U
        self.Delta = None
        self.neig = neig # used in diagonalization
        self.tol_fit = tol_fit
        self.max_fit = max_fit

    def fit(self, Delta):
        """Fit hybridization and update bath params."""
        #                __ n-1           2  
        #           !    \               Vk  
        # Delta(z)  =                 -------
        #                /__k = 0      z - ek
        n = self.n
        fopt = np.inf; it = 0
        while (fopt>self.tol_fit)&(it<self.max_fit):
            Delta_disc, fopt = fit_hybrid(Delta, n-1, self.nmats, self.beta, full_output=True)
            it += 1
        self.Delta = Delta_disc
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

    def Sigma(self, z):
        """Interacting self-energy."""
        #                  -1            -1
        #  Sigma(z)    =  g (z)     -   g (z)
        #                  0 
        return self.free(z, inverse=True) - np.reciprocal(self.gf(z.real, z.imag))

    def solve(self):
        """Solve impurity model and set interacting green's function."""
        H, V = self.H, self.V
        espace, egs = build_espace(H, V, self.neig)
        screen_espace(espace, egs)
        adjust_neigsector(espace, self.neig, self.n)
        self.gf = build_gf_lanczos(H, V, espace, self.beta, egs, repr='sp')
        self.espace = espace
        self.egs = egs

    # Helpers

    def integrate(self, mu=0.):
        # Up and Down have same spectrum
        return 2. * integrate_gf(self, mu)

    def __iter__(self):
        yield self


class SpinGfimp:
    """Spin dependent Green's function of SIAM model for FM solutions.

    The up and down green's functions point to the sub-Hamiltonians of this
    green's function, i.e. up points to H[0] and down to H[1]. When calling
    fit and/or update, the up & down green's function will fill their respective 
    H's matrix elements. However, the impurity model is solved simultaneously by
    passing H[:,:,:] and V[:,:] to build_espace and the spin dependent
    green's functions are built separately from the same espace.

    NOTE: Arrays of this class have the general shape = (2, z.size)
    """

    def __init__(self, n, nmats=3000, U=3., beta=1e6, neig=None, tol_fit=10., max_fit=3):
        self.nmats = nmats # Used in Matsubara fit. 
        self.beta = beta # Used in Matsubara fit and interacting green's function.
        self.n = n
        self.H = np.zeros((2,n,n))
        self.V = np.zeros((n,n))
        self.V[0,0] = U
        self.Delta = None
        self.neig = neig # used in diagonalization
        self.gfimp = [None, None]
        # Build gfimp's for each spin and make them point to self.H[spin].
        # This is a bad hack to avoid creating gfimp.H & gfimp.V since they
        # must point to the same (spin dependent) Hamiltonian and onsite
        # interaction.
        for s in range(2): 
            gfimp = object.__new__(Gfimp)
            gfimp.n = n
            gfimp.nmats = nmats
            gfimp.beta = beta
            gfimp.tol_fit = tol_fit
            gfimp.max_fit = max_fit
            gfimp.H = self.H[s]
            gfimp.V = self.V
            self.gfimp[s] = gfimp

    @property
    def mu(self):
        return -self.H[0,0,0]

    def fit(self, Delta):
        for gfimp, delta in zip(self, Delta):
            gfimp.fit(delta)

    def update(self, mu):
        self.H[:,0,0] = -mu
    
    def fit_update(self, Delta, mu):
        for gfimp, delta in zip(self, Delta):
            gfimp.fit_update(delta, mu)

    @property
    def up(self):
        return self.gfimp[0]

    @property
    def dw(self):
        return self.gfimp[1]

    def Sigma(self, z):
        return np.stack([gf.Sigma(z) for gf in self])

    def solve(self):
        """Solve impurity model."""
        H, V = self.H, self.V
        espace, egs = build_espace(H, V, self.neig)
        screen_espace(espace, egs)
        adjust_neigsector(espace, self.neig, self.n)
        for s, gfimp in enumerate(self):
            gfimp.gf = build_gf_lanczos(H, V, espace, self.beta, egs, repr='sp', ispin=s)
        self.espace = espace
        self.egs = egs

    def spin_symmetrize(self):
        self.dw.H[:] = self.up.H[:]
        self.dw.Delta = self.up.Delta

    # Helpers

    def integrate(self, mu=None):
        # Up and Down do NOT have same spectrum (in theory).
        return sum(integrate_gf(gfimp, mu) for gfimp in self)

    def __iter__(self):
        yield from self.gfimp


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


# def dmft_step(delta, gfimp, gfloc):
#     """Perform a DMFT self-consistency step."""
#     gfimp.fit(delta) # at matsubara frequencies
#     gfimp.solve()
#     gfloc.set_local(gfimp.Sigma)

def dmft_step(delta, gfimp, gfloc):
    """Perform a DMFT self-consistency step."""
    gfimp.fit(delta) # at matsubara frequencies
    gfimp.update(gfloc.mu-gfloc.ed)
    gfimp.solve()
    gfloc.set_local(gfimp.Sigma)


def dmft_step_adjust(delta, gfimp, gfloc, occupancy_goal):
    """Perform a DMFT self-consistency step and adjust chemical potential to 
    target occupation (for both local and impurity green's functions."""
    dmft_step(delta, gfimp, gfloc)
    # mu = adjust_mu(gfimp, occupancy_goal)
    mu = adjust_mu(gfloc, occupancy_goal)
    # gfimp.update(mu)
    gfloc.update(mu)


# def dmft_step_adjust_ext(delta, gfimp, gfloc):
#     """Set chemical potential and perform a DMFT self-consistency step.
#     The chemical potential is the first entry of the delta array and is
#     adjusted by an external minimizer during the self consistency loop."""
#     mu, delta = delta[0], delta[1:]
#     gfimp.update(mu)
#     gfloc.update(mu)
#     dmft_step(delta, gfimp, gfloc)


def dmft_step_magnetic(delta, gfimp, gfloc, sign, field):
    gfimp.up.fit(delta)
    gfimp.spin_symmetrize()
    gfimp.up.update(gfloc.mu+sign*field-gfloc.ed)
    gfimp.dw.update(gfloc.mu-sign*field-gfloc.ed)
    gfimp.solve()
    gfloc.set_local(gfimp.Sigma)


class DMFT:
    """Base class for DMFT self-consistent loop.
    
    Sub classes can overwrite the methods:
        - initialize
        - step
        - distance

    methods must:
        - initialize 
            call   : -
            return : initial guess
        - step 
            call   : -
            return : the current occupation and a new guess for the next iteration. 
        - distance 
            call   : __call__ 
            return : the error w.r.t. the previous iteration
    """
    
    def __init__(self, gfimp, gfloc, occupancy_goal=None, max_iter=20, tol=1e-3):
        self.gfimp = gfimp
        self.gfloc = gfloc
        self.occupancy_goal = occupancy_goal
        self.it = 0
        self.tol = tol
        self.max_iter = max_iter
        wn = (2*np.arange(gfimp.nmats)+1)*np.pi/gfimp.beta
        self.z = 1.j*wn

    def initialize(self, U, Sigma):
        mu = U/2.
        self.gfloc.set_local(Sigma)
        self.gfloc.update(mu)
        self.gfimp.update(mu)
        return self.gfloc.Delta(self.z)

    def initialize_magnetic(self, U, Sigma, sign, field):
        delta = self.initialize(U, Sigma)
        dmft_step_magnetic(delta, self.gfimp, self.gfloc, sign, field)
        return self.gfloc.Delta(self.z)

    def step(self, delta):
        dmft_step(delta, self.gfimp, self.gfloc)
        delta_new = self.gfloc.Delta(self.z)
        # occp = self.gfimp.integrate()
        occp = self.gfloc.integrate()
        return occp, delta_new

    def distance(self, delta):
        eps = self(delta) - delta
        return eps

    def __call__(self, delta):
        print(f'Iteration : {self.it:2}')
        non_causal = delta.imag>0 # ensures that the imaginary part is negative
        delta[non_causal].imag = -1e-20
        occp, delta_new = self.step(delta)
        print(f'Occupation : {occp:.5f} Chemical potential : {self.gfloc.mu:.5f}', end=' ')
        eps = np.linalg.norm(delta_new - delta)
        print(f'Error : {eps:.5f}')
        if eps < self.tol:
            raise Converged('Converged!')
        self.it += 1
        if self.it > self.max_iter:
            raise FailedToConverge('Failed to converge!')
        return delta_new

    def solve_with_broyden_mixing(self, delta, alpha=0.5, verbose=True, callback=None):
        broyden1(self.distance, delta, alpha=alpha, reduction_method="svd", 
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


# class _DMFTadjust(_DMFT):
#     """
#     As compare to the base DMFT class,
#     at each iteration step the chemical potential is adjusted
#     to obatin the target occupation, given the current set of 
#     parameters. 
    
#     NOTE: The impurity green's function is used to obatin 
#     the current occupation.
#     """
    
#     def step(self, delta):
#         dmft_step_adjust(delta, self.gfimp, self.gfloc, self.occupancy_goal)
#         delta_new = self.gfloc.Delta(self.z)
#         occp = self.gfimp.integrate()
#         return occp, delta_new


# class _DMFTadjustext(_DMFT):
#     """
#     As compared to the others, let the minimizer adjust
#     the chemical potential. The first entry of the variable 
#     `delta` contains the chemical potential. The first entry
#     in the `eps` variable contains the difference between
#     the current and the target occupation.
#     """
    
#     def initialize(self, U):
#         delta = np.empty(self.z.size+1, complex)
#         delta[1:] = super().initialize(U)
#         delta[0] = self.gfimp.mu
#         return delta

#     def step(self, delta):
#         dmft_step_adjust_ext(delta, self.gfimp, self.gfloc)
#         delta_new = np.empty_like(delta)
#         delta_new[1:] = self.gfloc.Delta(self.z)
#         delta_new[0] = self.gfimp.integrate()
#         return delta_new[0], delta_new

#     def distance(self, delta):
#         eps = self(delta) - delta
#         eps[0] = sum(delta[0] - self.occupancy_goal)
#         return eps


# def solve(gfimp, gfloc, occupancy_goal=None, max_iter=20, tol=1e-3, 
#           occp_method=None, mixing_method='broyden'):
    
#     if occp_method is None:
#         dmft_solver = _DMFT(gfimp, gfloc, max_iter=max_iter, tol=tol)
    
#     elif occp_method == 'adjust':
#         dmft_solver = _DMFTadjust(gfimp, gfloc, occupancy_goal, max_iter, tol)    
    
#     elif occp_method == 'adjust_ext':
#         dmft_solver = _DMFTadjustext(gfimp, gfloc, occupancy_goal, max_iter, tol)
    
#     else:
#         raise ValueError("Invalid occupation method. Choose between None, adjust, adjust_ext.")

#     delta = dmft_solver.initialize()
#     dmft_solver.solve(delta, mixing_method=mixing_method)