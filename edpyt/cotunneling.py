from numba import vectorize
import numpy as np
from itertools import zip_longest
from collections import defaultdict
from operator import attrgetter

from edpyt.integrals import Gamma1, Gamma2

from edpyt.sector import (
    OutOfHilbertError, get_c_sector, get_cdg_sector)

from edpyt.gf_exact import (
    # <N+1|c+(i,up)|N>
    # <N|c-(i,up)|N+1>
    project_exact_up, 
    # <N+1|c+(i,dw)|N>
    # <N|c-(i,dw)|N+1>
    project_exact_dw)

from edpyt.operators import (
    cdg, c)

@vectorize('float64(float64, float64)')
def abs2(r, i):
    return r**2 + i**2


def projector(ispin):
    if ispin==0:
        return project_exact_up
    return project_exact_dw


def project_add(ispin_i, ispin_j, n, nupI, ndwI, espace):
    """Cotunneling rate for a lead electron to go from lead l' to lead l and
    the central region to go from state n' to state n, i.e.:
     
            l' -> l
            n' -> n     
    """
    #            ___                           
    #   +        \                                            +
    #  y (ss') =        A     A       < n| c    | m+ > < m+| c      | n' >
    #   nn'      /__     l,i    l',i'        i,s               i',s'
    #            ii'                            
    nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin_j) # |m+>
    nupF, ndwF = get_c_sector(nupJ, ndwJ, ispin_i) # |n>
    sctI = espace[(nupI, ndwI)] # |n'>
    sctJ = espace[(nupJ, ndwJ)] # |m+>
    sctF = espace[(nupF, ndwF)] # |n>
    proj_ispin_j = projector(ispin_j) 
    proj_ispin_i = projector(ispin_i) 
    # Because one assumes that only the charge is a conserved quantity
    # (and not Sz), the states are product of up and down comonponents: 
    # 
    #    |s> = |s>   ,  |s> 
    #             dw      up
    #
    # e.g.
    #
    #   |0,0,0,1,...1,0>, |0,1,0,1,...0,,0>
    #
    # Hence, one should include the fermionic sign coming from the number
    # of down spins when the operator acts on the up component.
    sgnI = (-1.)**ndwI if ispin_j==0 else 1. 
    sgnJ = (-1.)**ndwJ if ispin_i==0 else 1.
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    v_FJ = np.empty((sctF.eigvals.size,sctJ.eigvals.size,n))
    for j in range(n):
        v_JI[...,j] = sgnI * proj_ispin_j(j, cdg, sctI, sctJ)
    for i in range(n):
        v_FJ[...,i] = sgnJ * proj_ispin_i(i, c, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = sctJ.eigvals[None,:]-sctI.eigvals[:,None]
    gf2elist = []
    for f,i in np.ndindex(v_FJ.shape[0],v_JI.shape[1]):
        gf2elist.append(Gf2((nupF,ndwF,f),(nupI,ndwI,i),v_FJ[f],v_JI[:,i].copy(),E[i],dE[f,i],(nupJ,ndwJ)))
    return gf2elist


def project_sub(ispin_i, ispin_j, n, nupI, ndwI, espace):
    """Cotunneling rate for a lead electron to go from lead l to lead l' and
    the central region to go from state n' to state n, i.e.:
    
            l' -> l
            n' -> n    
    """
    #            ___                           
    #   -        \                           +                 
    #  y (ss') =        A     A       < n| c    | m- > < m-| c      | n' >
    #   nn'      /__     l',i    l,i'        i,s               i',s'
    #            ii'                           
    nupJ, ndwJ = get_c_sector(nupI, ndwI, ispin_j) # |m+>
    nupF, ndwF = get_cdg_sector(n, nupJ, ndwJ, ispin_i) # |n>
    sctI = espace[(nupI, ndwI)] # |n'>
    sctJ = espace[(nupJ, ndwJ)] # |m+>
    sctF = espace[(nupF, ndwF)] # |n>
    proj_ispin_j = projector(ispin_j) 
    proj_ispin_i = projector(ispin_i) 
    # See notes for add projection.
    sgnI = (-1.)**ndwI if ispin_j==0 else 1.
    sgnJ = (-1.)**ndwJ if ispin_i==0 else 1.
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    v_FJ = np.empty((sctF.eigvals.size,sctJ.eigvals.size,n))
    for j in range(n):
        v_JI[...,j] = sgnI * proj_ispin_j(j, c, sctI, sctJ)
    for i in range(n):
        v_FJ[...,i] = sgnJ * proj_ispin_i(i, cdg, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = -sctJ.eigvals[None,:]+sctI.eigvals[:,None]
    gf2hlist = []
    for f,i in np.ndindex(v_FJ.shape[0],v_JI.shape[1]):
        gf2hlist.append(Gf2((nupF,ndwF,f),(nupI,ndwI,i),v_FJ[f],v_JI[:,i].copy(),E[i],dE[f,i],(nupJ,ndwJ)))
    return gf2hlist


def project_sector(n, nupI, ndwI, espace, sigmadict, ispin=None):
    """Cotunneling rate from state n to state n'.

    NOTE: The leads' and spin's indices are interchanged. For leads' indices
    this is because we want the sum l' -> l + l' <- l. For spin, one can show
    that to obtain the same final state n, the spin creation and annihilation
    orders have to be swaped.
    
    Example:

    |n'> = u,d
    |n> = u+1,d-1
    
        |    u+     d-   |    d-      u+
    ---------------------|-----------------
    u,d | u+1,d  u+1,d-1 |  u,d-1  (u+1,d-1)   , same final state   
    """
    #                                            
    #                                                      
    #  __ll'ss'             +            1                       -               1      
    #  \        (E)  =  (  y   (ss') ----------------     +     y   (s's) ----------------   ) n(E - (E - E)  - u)  (1-  n(E - u))
    #  /__                  nn'       E -( E   - E  )            nn'      E - (E   - E)                n   n'    L              R     
    #     nn'                               m+    n'                             n'    m-
    #                                                                                                                           
    if ispin is None: 
        ispin = [(0,0),(0,1),(1,0),(1,1)]
    sigmalist = [] #dict.fromkeys(ispin)
    for ispin_i, ispin_j in ispin:
        # Electron and hole green functions.
        try:
            gf2elist = project_add(ispin_i, ispin_j, n, nupI, ndwI, espace)
        except OutOfHilbertError:
            gf2elist = (None,)
        try:
            gf2hlist = project_sub(ispin_j, ispin_i, n, nupI, ndwI, espace)
        except OutOfHilbertError:
            gf2hlist = (None,)
        sigmalist.extend([Sigma(gf2e,gf2h)
                          for gf2e,gf2h in zip_longest(gf2elist,gf2hlist)])
    for sigma in sigmalist:
        sigmadict[(sigma.idF,sigma.idI)].append(sigma)
    return sigmadict


def build_transition_elements(n, egs, espace, cutoff=None):
    """Build projector space. Compute all possible matrix elements connecting 
    the ground state and the vectors reached by the latter (including the matrix
    elements connecting the latter vectors).
    """
    ispin = [(0,0),(0,1),(1,0),(1,1)]
    dS = [(0,0),   # add up remove up
          (-1,1),  # add dw remove up
          (1,-1),  # remove dw add up
          (0,0)]   # add dw remove dw
    ngs = [ns for ns,sct in espace.items() if abs(sct.eigvals.min()-egs)<1e-9]
    reached_by_gs = [(ns[0]+ds[0],ns[1]+ds[1]) for ns in ngs for ds in dS]
    # Loop over the sectors reached by the GS sector and compute the
    # projections to sectors that are also reached by the GS. Note that
    # this ensures that the (self) matrix elements bringing to the same sector
    # are also included.
    sigmadict = defaultdict(list)
    for ns in np.unique(reached_by_gs,axis=0):
        ispin_subset = [i for i,ds in zip(ispin,dS) if (ns[0]+ds[0],ns[1]+ds[1]) in reached_by_gs]
        project_sector(n, ns[0], ns[1], espace, sigmadict, ispin_subset)
    if cutoff is not None:
        return screen_transition_elements(sigmadict, egs, espace, cutoff)
    return sigmadict


class Gf2:
    """Green's function.
    
    Args:
        E : (E+ - En') or (En' - E-)
        v_FJ : <n|c+|m-> or <n|c|m+>
        v_JI : <m-|c|n'> or <m+|c+|n'>
        dE = En-En'
        dS = change in spin.
    """
                           
    #    +(-)              1           
    #  y          ---------------- 
    #    nn'       E -( E   - E  )
    #                    m+    n' 
    def __init__(self, idF, idI, vF, vI, E, dE, nsJ) -> None:
        self.idF = idF
        self.idI = idI
        self.vF = vF
        self.vI = vI
        self.E = E
        self.dE = dE
        self.dS = (idF[0]-idI[0],idF[1]-idI[1])
        self.nsJ = nsJ
        
    def __call__(self, z, aF, aI):
        z = np.atleast_1d(z)
        res = np.einsum('j,j,kj->k',aF.dot(self.vF.T),aI.dot(self.vI.T),
                        np.reciprocal(z-self.E[None,:]))
        return res


class _Sigma:
    """Base class for Sigma."""
    def __init__(self, gf2) -> None:
        self.gf2 = gf2

    def __getattr__(self, name):
        if name in ['idF','idI','dS','dE']:
            return getattr(self.gf2, name)
        raise AttributeError

    @staticmethod
    def G(beta, z):
        """Helper function for approximate solution."""
        if (beta*abs(z))<1e-18:
            return 1.
        if (beta*z)>1e3:
            return 0.
        return z/(np.exp(beta*z)-1)

    def approximate(self, A, extract, inject, beta, mu):
        return self.G(beta, self.dE-mu[extract]+mu[inject]
               ) * self(0.5*(self.dE+sum(mu)), A, extract, inject)
              

class _Sigmae(_Sigma):
    """Electron."""
    #   |        |  2
    #   | G  (z) |      
    #   |  e     |    
    
    def integrate(self, A, extract, inject, beta, mu):
        return Gamma1(
            A[inject].dot(self.gf2.vF[0])*A[extract].dot(self.gf2.vI[0]), # A
            self.gf2.E[0], # epsA
            [mu[extract]-self.dE,mu[inject]], beta)

    def __call__(self, z, A, extract, inject):
        res = self.gf2(z, A[inject], A[extract])
        return abs2(res.real, res.imag)
    

class _Sigmah(_Sigma):
    """Hole."""
    #   |        |  2
    #   | G  (z) |      
    #   |  h     |    
    
    def integrate(self, A, extract, inject, beta, mu):
        return Gamma1(
            A[extract].dot(self.gf2.vF[0])*A[inject].dot(self.gf2.vI[0]), # A
            self.gf2.E[0], # epsA
            [mu[extract]-self.dE,mu[inject]], beta)
    
    def __call__(self, z, A, extract, inject):
        res = self.gf2(z, A[extract], A[inject])
        return abs2(res.real, res.imag)
     
               
class Sigma(_Sigma):
    #   |                   |  2
    #   | G  (z) + G    (z) |      
    #   |  e         h      |      
    def __new__(cls, gf2e, gf2h):
        if gf2h is None:
            return _Sigmae(gf2e)
        if gf2e is None:
            return _Sigmah(gf2h)
        return super().__new__(cls)
    
    def __init__(self, gf2e, gf2h) -> None:
        assert np.allclose(gf2e.idF, gf2h.idF)&np.allclose(gf2e.idI, gf2h.idI), """
                Invalid projections. Initial and final states 
                after N+1 and N-1 projections are not the same."""
        self.gf2e = gf2e
        self.gf2h = gf2h
        # Default of gf2e params
        super().__init__(self.gf2e)
    
    def integrate(self, A, extract, inject, beta, mu):
        return Gamma2(
            A[inject].dot(self.gf2e.vF[0])*A[extract].dot(self.gf2e.vI[0]), #A
            A[extract].dot(self.gf2h.vF[0])*A[inject].dot(self.gf2h.vI[0]), #B
            self.gf2e.E[0], # epsA
            self.gf2h.E[0], # epsB
            [mu[extract]-self.dE,mu[inject]], beta)

    def __call__(self, z, A, extract, inject):
        res = self.gf2e(z, A[inject], A[extract]) + self.gf2h(z, A[extract], A[inject])
        return abs2(res.real, res.imag)


def build_rate_matrix(sigmadict, beta, mu, A, approx_integral=False):
    #          ___
    #         |     __                      
    #         |    \     _             _   
    #         |  -      |             |            --  
    #         |    /__     k1           12
    #         |        k!=1           __         
    #  W  =   |        _             \     _       --  
    #         |       |            -      |          
    #         |         21           /__     k2  
    #         |                          k!=2
    #         |       :                :          \
    integrate = attrgetter('approximate') if approx_integral else attrgetter('integrate')
    sz, odd = np.divmod(len(sigmadict), 2)
    assert ~odd, """
        Invalid sigma list. Each matrix element must contain 
        its complex conjugate.
    """
    idF = sorted(set([idF for idF, _ in sigmadict.keys()]))
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    W = np.zeros((sz, sz))
    for extract, inject in np.ndindex(2, 2):
        for i, f in np.ndindex(sz, sz):
            if i != f:
                try:
                    gamma = 0.
                    for sigma in sigmadict[map[f],map[i]]:
                        gamma += integrate(sigma)(A, extract, inject, beta, mu)
                except:
                    continue
                W[i,i] -= gamma
                W[f,i] += gamma
    return W


def build_transition_matrix(sigmadict, beta, mu, A, extract, inject, approx_integral=False):
    sz, odd = np.divmod(len(sigmadict), 2)
    assert ~odd, """
        Invalid sigma list. Each matrix element must contain 
        its complex conjugate.
    """
    integrate = attrgetter('approximate') if approx_integral else attrgetter('integrate')
    idF = sorted(set([idF for idF, _ in sigmadict.keys()]))
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    T = np.zeros((sz, sz))
    for i, f in np.ndindex(sz, sz):
        try:
            gamma = 0.
            for sigma in sigmadict[map[f],map[i]]:
                gamma += integrate(sigma)(A, extract, inject, beta, mu)
                gamma -= integrate(sigma)(A, inject, extract, beta, mu)
        except:
            continue
        T[f,i] = gamma
    return T


def screen_transition_elements(sigmadict, egs, espace, cutoff):
    return {(idF, idI):sigma for (idF, idI),sigma in sigmadict.items() 
            if (abs(sigma[0].dE)<cutoff)
            and(abs(espace[idF[:2]].eigvals[idF[2]]-egs)<cutoff)
            and(abs(espace[idI[:2]].eigvals[idI[2]]-egs)<cutoff)}


def find_active_sectors(ngs):
    """Find all sectors that are necessary to diagonalize in order
    to build the cotunniling transition and rate matrices.
    Args:
        ngs : (list of tuples)
            list of tuples with the ground state sectors,
            e.g. [(nup,ndw),(nup,ndw),....]
    """
    dS = [(0,0),(-1,1),(1,-1)]
    dN = [(0,1),(0,-1),(1,0),(-1,0)]
    reached_by_gs = [(ns[0]+ds[0],ns[1]+ds[1]) for ns in ngs for ds in dS]
    active_sectors = [ns for ns in reached_by_gs]
    for ns in np.unique(reached_by_gs,axis=0):
        active_sectors.extend([(ns[0]+dn[0],ns[1]+dn[1]) for ds in dS for dn in dN 
                               if (ns[0]+ds[0],ns[1]+ds[1]) in reached_by_gs])
    return sorted([tuple(ns) for ns in set(active_sectors)])


def get_active_neig(n, ngs, val=2):
    from edpyt.lookup import get_sector_index
    neig = np.zeros((n+1)*(n+1),int)
    idx = [get_sector_index(n,nup,ndw) for nup,ndw in find_active_sectors(ngs)]
    for i in idx:
        neig[i] = val
    return neig