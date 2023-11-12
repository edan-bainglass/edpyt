from warnings import warn

import numpy as np
from scipy.optimize import broyden1, linearmixing, root_scalar

# from edpyt_backend.fit_wrap import fit_hybrid
from edpyt.espace import adjust_neigsector, build_espace, screen_espace
# from edpyt.fit import Delta, set_initial_bath
from edpyt.fit_cg import _delta, fit_hybrid, get_initial_bath
from edpyt.gf_lanczos import build_gf_lanczos
from edpyt.integrate_gf import integrate_gf


def adjust_mu(gf, occupancy_goal, bracket=(-20.0, 20)):
    """Get the chemical potential to obtain the occupancy goal.

    NOTE : The gf is supposed to have the general form
    (z+mu-Delta(z)-Sigma(z))^-1. Here, `distance` returns
    the change in mu required to satisfy the occupancy goal.
    """
    # distance = lambda mu: np.sum(gf.integrate(mu)-occupancy_goal)
    distance = lambda mu: gf.integrate(mu).sum() - occupancy_goal.sum()
    return root_scalar(distance, bracket=bracket,
                       method="brentq").root  # + gf.mu


class Converged(Exception):

    def __init__(self, message):
        self.message = message


class FailedToConverge(Exception):

    def __init__(self, message):
        self.message = message


class Delta:

    def __init__(self, nbath, nmats, beta) -> None:
        self.nbath = nbath
        self.nmats = nmats
        self.beta = beta
        self.x = np.empty(2 * nbath)
        self.reset_bath()

    def reset_bath(self):
        get_initial_bath(p=self.x)

    @property
    def ek(self):
        return self.x[:self.nbath]

    @property
    def vk(self):
        return self.x[self.nbath:]

    def fit(self, delta):
        fit_hybrid(self.x, self.nmats, delta, self.beta)

    def __call__(self, z):
        return _delta(self.x, z)


class Gfimp:
    """Green's function of SIAM model.

    Args:
        nmats : # of matzubara frequencies used to fit the hybridization.
        U : impurity local interaction.
        beta : 1/kBT
        neig : # of eigenvaleus to solve per sector. Defaults to None (solve all)
        tol_fit : max. error above which the fit is repeated.
        max_fit : max. fit repetions.
        alpha : weigth matzubara frequencies.
    """

    # Matrix form
    #
    #   | -mu  v0 v1 . . |
    #   |  v0  e0        |
    #   |  v1     e1     |
    #   |   .        .   |
    #   |   .          . |
    def __init__(self,
                 n,
                 nmats=3000,
                 U=3.0,
                 beta=1e6,
                 neig=None,
                 adjust_neig=False,
                 spin=0):
        self.n = n
        self.Delta = Delta(n - 1, nmats, beta)
        self.spin = spin
        self.H = np.zeros((n, n))
        self.V = np.zeros((n, n))
        self.V[0, 0] = U
        self.neig = neig  # used in diagonalization.
        self.adjust_neig = (
            adjust_neig  # adjust # of eigenvalues to solve after each solution.
        )

    @property
    def nmats(self):
        if hasattr(self, "Delta"):
            return self.Delta.nmats

    @property
    def beta(self):
        if hasattr(self, "Delta"):
            return self.Delta.beta

    @property
    def x(self):
        if hasattr(self, "Delta"):
            return self.Delta.x

    def reset_bath(self):
        self.Delta.reset_bath()

    @property
    def mu(self):
        # ed = -mu
        return -self.H[0, 0]

    def fit(self, delta):
        """Fit hybridization and update bath params."""
        #                __ n-1           2
        #           !    \               Vk
        # Delta(z)  =                 -------
        #                /__k = 0      z - ek
        self.Delta.fit(delta)
        self.H[1:, 0] = self.H[0, 1:] = self.Delta.vk
        self.H.flat[(self.n + 1)::(self.n + 1)] = self.Delta.ek

    def update(self, mu):
        """Update chemical potential."""
        self.H[0, 0] = -mu

    def fit_update(self, Delta, mu):
        """Update impurity model."""
        self.fit(Delta)
        self.update(mu)

    def free(self, z, inverse=False):
        """Non-interacting green's function."""
        #                 1
        # g  =    -----------------
        #         z + mu - Delta(z)
        g0_inv = z - self.H[0, 0] - self.Delta(z)
        if inverse:
            return g0_inv
        return np.reciprocal(g0_inv)

    def __call__(self, z):
        """Interacting green's function."""
        #                    1
        # g  =    ----------------------------
        #  0      z + mu - Delta(z) - Sigma(z)
        return np.reciprocal(z - self.H[0, 0] - self.Delta(z) - self.Sigma(z))

    def Sigma(self, z):
        """Interacting self-energy."""
        #                  -1            -1
        #  Sigma(z)    =  g (z)     -   g (z)
        #                  0
        return self.free(z, inverse=True) - np.reciprocal(
            self.gf(z.real, z.imag))

    def solve(self):
        """Solve impurity model and set interacting green's function."""
        H, V = self.H, self.V
        espace, egs = build_espace(H, V, self.neig)
        screen_espace(espace, egs)  # , beta=self.beta)
        if self.adjust_neig:
            adjust_neigsector(espace, self.neig, self.n)
        self.gf = build_gf_lanczos(H,
                                   V,
                                   espace,
                                   self.beta,
                                   egs,
                                   repr="sp",
                                   ispin=self.spin)
        self.espace = espace
        self.egs = egs

    # Helpers

    def integrate(self, mu=0.0):
        # Up and Down have same spectrum
        return 2.0 * integrate_gf(self, mu)

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

    def __init__(self,
                 n,
                 nmats=3000,
                 U=3.0,
                 beta=1e6,
                 neig=None,
                 adjust_neig=False):
        self.H = np.zeros((2, n, n))
        self.gfimp = [None, None]
        # Build gfimp's for each spin and make them point to self.H[spin].
        # This is a bad hack to avoid creating gfimp.H & gfimp.V since they
        # must point to the same (spin dependent) Hamiltonian and onsite
        # interaction.
        for s in range(2):
            gfimp = Gfimp(n, nmats, U, beta, neig, adjust_neig, spin=s)
            gfimp.H = self.H[s]
            self.gfimp[s] = gfimp

    def __getattr__(self, name):
        """Search in up green's function for attribute."""
        return getattr(self.gfimp[0], name)

    def reset_bath(self):
        for gf in self:
            gf.reset_bath()

    @property
    def mu(self):
        return -self.H[0, 0, 0]

    def fit(self, delta):
        for i, gfimp in enumerate(self):
            gfimp.fit(delta[i])

    def update(self, mu):
        self.H[:, 0, 0] = -mu

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
        for gf in self:
            gf.solve()

    def solve(self):
        """Solve impurity model and set interacting green's function."""
        H, V = self.H, self.V
        espace, egs = build_espace(H, V, self.neig)
        screen_espace(espace, egs)
        if self.adjust_neig:
            adjust_neigsector(espace, self.neig, self.n)
        for gf in self:
            gf.gf = build_gf_lanczos(H,
                                     V,
                                     espace,
                                     self.beta,
                                     egs,
                                     repr="sp",
                                     ispin=gf.spin)
        self.espace = espace
        self.egs = egs

    def spin_symmetrize(self):
        self.dw.H[:] = self.up.H[:]
        self.dw.Delta.x[:] = self.up.Delta.x[:]

    # Helpers

    def integrate(self, mu=None):
        # Up and Down do NOT have same spectrum (in theory).
        return sum(integrate_gf(gfimp, mu) for gfimp in self)

    def __iter__(self):
        yield from self.gfimp

    def __getitem__(self, i):
        return self.gfimp[i]


class Gfloc:
    """Parent local green's function.
    """

    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to local self-energy!"""
        self.Sigma = Sigma

    def Delta(self, z: complex) -> complex:
        """Calculates local hybridization according to Eq. 1.74 in Guido's thesis.
        .. math::
        \Delta(z) = z + \mu - \Sigma(z) - G(z)^{-1}

        Parameters
        ----------
        `z` : `complex`
            #UNKNOWN Energy but in the complex plane?

        Returns
        -------
        `complex`
        .. math::
        \Delta(z)
        """
        return z + self.mu - self.Weiss(z)

    def Weiss(self, z: complex) -> complex:
        """Calculates the Weiss field given by

        .. math::
        \Sigma(z) + G(z)^{-1}

        Parameters
        ----------
        `z` : `complex`
            #UNKNOWN Energy but in the complex plane?

        Returns
        -------
        `complex`
           Weiss field
        """
        gloc_inv = self(z, inverse=True)
        return self.Sigma(z) + gloc_inv


# Analytical Bethe lattice
_ht = lambda z: 2 * (z - 1j * np.sign(z.imag) * np.sqrt(1 - z**2))
eps = 1e-20
ht = lambda z: _ht(z.real + 1.0j * (z.imag if z.imag > 0.0 else eps))


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
        gloc_inv = self.free(z, inverse=True) - self.Sigma(z)
        if inverse:
            return gloc_inv
        return np.reciprocal(gloc_inv)

    def free(self, z, inverse=False):
        g0inv = z + self.mu - self.Hybrid(z)
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
        gloc = self.hilbert(z + self.mu - self.Sigma(z))
        if inverse:
            return np.reciprocal(gloc)
        return gloc

    def free(self, z, inverse=False):
        g0 = self.hilbert(z + self.mu)
        if inverse:
            return np.reciprocal(g0)
        return g0


def dmft_step(delta, gfimp, gfloc):
    """Perform a DMFT self-consistency step."""
    gfimp.fit(delta)  # at matsubara frequencies
    gfimp.update(gfloc.mu - gfloc.ed)
    gfimp.solve()
    gfloc.set_local(gfimp.Sigma)


def dmft_step_adjust(delta, gfimp, gfloc, occupancy_goal):
    """Perform a DMFT self-consistency step and adjust chemical potential to
    target occupation (for both local and impurity green's functions."""
    dmft_step(delta, gfimp, gfloc)
    mu = adjust_mu(gfloc, occupancy_goal)
    gfloc.update(mu)


def dmft_step_magnetic(delta, gfimp, gfloc, sign, field):
    gfimp.up.fit(delta)
    gfimp.spin_symmetrize()
    gfimp.up.update(gfloc.mu + sign * field - gfloc.ed)
    gfimp.dw.update(gfloc.mu - sign * field - gfloc.ed)
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

    def __init__(
        self,
        gfimp,
        gfloc,
        occupancy_goal=None,
        max_iter: int = 20,
        tol: float = 1e-3,
        adjust_mu: bool = False,
        alpha: float = 0.0,
    ):
        self.gfimp = gfimp
        self.gfloc = gfloc
        self.occupancy_goal = occupancy_goal
        self.it = 0
        self.tol = tol
        self.max_iter = max_iter
        wn = (2 * np.arange(gfimp.nmats) + 1) * np.pi / gfimp.beta
        self.z = 1.0j * wn
        self.delta = None
        self.adjust_mu = adjust_mu
        self.weights = wn**-alpha

    def initialize(self, U, Sigma, mu=None):
        if mu is None:
            mu = U / 2.0
        self.gfloc.set_local(Sigma)
        self.gfloc.update(mu)
        self.gfimp.update(mu - self.gfloc.ed)
        self.gfimp.reset_bath()
        return self.gfloc.Delta(self.z)

    def initialize_magnetic(self, U, Sigma, sign, field, mu=None):
        delta = self.initialize(U, Sigma, mu)
        dmft_step_magnetic(delta, self.gfimp, self.gfloc, sign, field)
        return self.gfloc.Delta(self.z)

    def step(self, delta):
        if self.adjust_mu:
            dmft_step_adjust(delta, self.gfimp, self.gfloc,
                             self.occupancy_goal)
            occp = self.gfloc.integrate(self.gfloc.mu)
        else:
            dmft_step(delta, self.gfimp, self.gfloc)
            occp = self.occupancy_goal
        delta_new = self.gfloc.Delta(self.z)
        return np.sum(occp), delta_new

    def distance(self, delta):
        eps = self.weights * (self(delta) - delta)
        return eps

    def __call__(self, delta):
        print(f"Iteration : {self.it:2}")
        self.delta = delta
        non_causal = delta.imag > 0  # ensures that the imaginary part is negative
        delta[non_causal].imag = -1e-20
        occp, delta_new = self.step(delta)
        print(
            f"Occupation : {occp:.5f} Chemical potential : {self.gfloc.mu:.5f}",
            end=" ")
        eps = np.linalg.norm(delta_new - delta)
        print(f"Error : {eps:.5f}")
        if eps < self.tol:
            raise Converged("Converged!")
        self.it += 1
        if self.it >= self.max_iter:
            raise FailedToConverge("Failed to converge!")
        return delta_new

    def solve_with_broyden_mixing(self,
                                  delta,
                                  alpha=0.5,
                                  verbose=True,
                                  callback=None):
        broyden1(
            self.distance,
            delta,
            alpha=alpha,
            reduction_method="svd",
            max_rank=10,
            verbose=verbose,
            f_tol=1e-99,
            callback=callback,
        )  # Loop forever (small f_tol!)

    def solve_with_linear_mixing(self,
                                 delta,
                                 iter=60,
                                 alpha=0.2,
                                 callback=None):
        linearmixing(self.distance,
                     delta,
                     iter=iter,
                     alpha=alpha,
                     callback=callback)

    def solve(self, delta, mixing_method="broyden", **kwargs):
        """'linear' or 'broyden' mixing
        the quantity being mixed is the hybridisation function

        """
        try:
            if mixing_method == "linear":
                self.solve_with_linear_mixing(delta, **kwargs)
            elif mixing_method == "broyden":
                self.solve_with_broyden_mixing(delta, **kwargs)
        except Converged:
            pass
        except FailedToConverge as err:
            print(err)
        # finally:
        #     np.save("data_DELTA_DMFT.npy", self.delta)
