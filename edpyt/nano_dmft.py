import numpy as np

from edpyt.integrate_gf import integrate_gf
# from edpyt.dmft import _DMFT, adjust_mu


def _get_sigma_method(comm):
    if comm is not None:
        def wrap(self, z):
            # Collect sigmas and expand to non equivalent indices.
            comm = self.comm
            sigma_loc = self.Sigma(z)
            sigma = np.empty(comm.size*sigma_loc.size, sigma_loc.dtype)
            comm.Allgather([sigma_loc, sigma_loc.size], [sigma, sigma_loc.size])
            shape = list(sigma_loc.shape)
            shape[0] *= comm.size
            return sigma.reshape(shape)
    else:
        def wrap(self, z):
            return self.Sigma(z)
    return wrap


def _get_idx_world(comm, n):
    if comm is None:
        return slice(None)
    else:
        stride = n//comm.size
        return slice(comm.rank*stride,(comm.rank+1)*stride)


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
    def __init__(self, H, S, Hybrid, idx_neq, idx_inv, comm=None) -> None:
        self.n = H.shape[-1]
        self.H = H
        self.S = S
        self.Hybrid = Hybrid
        self.idx_neq = idx_neq[_get_idx_world(comm, len(idx_neq))]
        self.idx_inv = idx_inv
        self.comm = comm
        self.get_sigma = _get_sigma_method(comm).__get__(self)

    @property
    def ed(self):
        return self.H.diagonal()[self.idx_neq]

    def __call__(self, z):
        """Interacting Green's function."""
        z = np.atleast_1d(z)
        sigma = self.get_sigma(z)
        it = np.nditer([sigma, z, None], flags=['external_loop'], order='F')
        with it:
            for sigma_, z_, out in it:
                x = self.free(z_[0], inverse=True)
                x.flat[::(self.n+1)] -= sigma_[self.idx_inv]
                out[...] = np.linalg.inv(x).diagonal()[self.idx_neq]
            return it.operands[2]

    def update(self, mu):
        """Update chemical potential."""
        self.mu = mu

    def set_local(self, Sigma):
        """Set impurity self-energy to diagonal elements of local self-energy!"""
        #
        # TODO : actually compute and store sigma.
        #
        self.Sigma = Sigma

    def Delta(self, z):
        """Hybridization."""
        #                                       -1
        # Delta(z) = z+mu - Sigma(z) - ( G (z) )
        #                                 ii
        z = np.atleast_1d(z)
        weiss = self.Weiss(z)
        ndim = weiss.ndim-1
        it = np.nditer([self.ed, weiss, z, None], 
                        op_axes=[[0]+[-1]*ndim,None,[-1]*ndim+[0],None], 
                        flags=['external_loop'], order='F')
        with it:
            for ed, weiss, z, out in it:
                out[...] = z+self.mu-ed-weiss
            return it.operands[3]

    def Weiss(self, z):
        """Weiss field."""
        #  -1                             -1
        # G     (z) = Sigma(z) + ( G (z) )
        #  0,ii                     ii
        gloc_inv = np.reciprocal(self(z))
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

    # Helper

    def integrate(self, mu=0.):
        occps = np.squeeze(integrate_gf(self, mu)[self.idx_inv,...])
        if occps.ndim<2:
            return 2. * occps.sum()
        return occps.sum()


class Gfimp:

    def __init__(self, gfimp) -> None:
        self.gfimp = gfimp

    def __getattr__(self, name):
        """Default is to return attribute of first impurity."""
        return getattr(self.gfimp[0], name)

    def update(self, mu):
        """Updated chemical potential."""
        mu = np.broadcast_to(mu, len(self))
        for i, gf in enumerate(self):
            gf.update(mu[i])

    @property
    def up(self):
        return Gfimp([gf.up for gf in self])

    @property
    def dw(self):
        return Gfimp([gf.dw for gf in self])

    def fit(self, delta):
        """Fit discrete bath."""
        for i, gf in enumerate(self):
            gf.fit(delta[i,...])

    def Sigma(self, z):
        """Correlated self-energy."""
        return np.stack([gf.Sigma(z) for gf in self])

    def solve(self):
        for gf in self:
            gf.solve()

    def spin_symmetrize(self):
        for gf in self:
            gf.spin_symmetrize()

    def __getitem__(self, i):
        return self.gfimp[i]

    def __len__(self):
        return len(self.gfimp)

    def __iter__(self):
        yield from iter(self.gfimp)


# def dmft_step(delta, gfimp, gfloc):
#     """Perform a DMFT self-consistency step."""
#     for i, gf in enumerate(gfimp):
#         gf.fit(delta[i,...])
#         gf.update(gfloc.mu-gfloc.ed[i])
#         gf.solve()
#     gfloc.set_local(gfimp.Sigma)


# def dmft_step_adjust(delta, gfimp, gfloc, occupancy_goal):
#     """Perform a DMFT self-consistency step and adjust chemical potential to 
#     target occupation (for both local and impurity green's functions."""
#     dmft_step(delta, gfimp, gfloc)
#     mu = adjust_mu(gfloc, occupancy_goal)
#     gfloc.update(mu)


# def dmft_step_adjust_ext(delta, gfimp, gfloc):
#     """Set chemical potential and perform a DMFT self-consistency step.
#     The chemical potential is the first entry of the delta array and is
#     adjusted by an external minimizer during the self consistency loop."""
#     mu, delta = delta[0], delta[1:].reshape(-1,len(gfimp))
#     gfimp.update(mu)
#     gfloc.update(mu)
#     dmft_step(delta, gfimp, gfloc)


# class _nanoDMFT(_DMFT):

#     def step(self, delta):
#         dmft_step(delta, self.gfimp, self.gfloc)
#         delta_new = self.gfloc.Delta(self.z)
#         occp = sum(self.gfloc.integrate().flat[self.gfloc.idx_inv])
#         return occp, delta_new


# class _nanoDMFTadjust(_DMFT):

#     def step(self, delta):
#         dmft_step_adjust(delta, self.gfimp, self.gfloc, self.occupancy_goal)
#         delta_new = self.gfloc.Delta(self.z)
#         occp = sum(self.gfloc.integrate().flat[self.gfloc.idx_inv])
#         return occp, delta_new


# class _nanoDMFTadjustext(_DMFT):

#     def initialize(self, U):
#         delta = np.empty(1+self.z.size*len(self.gfimp), complex)
#         delta[1:] = super().initialize(U).flat
#         delta[0] = self.gfimp.mu
#         return delta

#     def step(self, delta):
#         dmft_step_adjust_ext(delta, self.gfimp, self.gfloc)
#         delta_new = np.empty_like(delta)
#         delta_new[1:] = self.gfloc.Delta(self.z)
#         delta_new[0] = sum(self.gfimp.integrate())
#         return delta_new[0], delta_new

#     def distance(self, delta):
#         eps = self(delta) - delta
#         eps[0] = sum(delta[0] - self.occupancy_goal)
#         return eps


# def solve(gfimp, gfloc, occupancy_goal=None, max_iter=20, tol=1e-3, 
#           occp_method=None, mixing_method='broyden'):
    
#     if occp_method is None:
#         dmft_solver = _nanoDMFT(gfimp, gfloc, max_iter=max_iter, tol=tol)
    
#     elif occp_method == 'adjust':
#         dmft_solver = _nanoDMFTadjust(gfimp, gfloc, occupancy_goal, max_iter, tol)    
    
#     elif occp_method == 'adjust_ext':
#         dmft_solver = _nanoDMFTadjustext(gfimp, gfloc, occupancy_goal, max_iter, tol)
    
#     else:
#         raise ValueError("Invalid occupation method. Choose between None, adjust, adjust_ext.")

#     delta = dmft_solver.initialize()
#     dmft_solver.solve(delta, mixing_method=mixing_method)