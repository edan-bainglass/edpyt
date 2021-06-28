from numpy.lib.function_base import delete
from edpyt.integrals import Gamma1, Gamma2
from numba import vectorize
import numpy as np
from itertools import zip_longest
from collections import defaultdict

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
    check_empty as not_empty, 
    check_full as not_full,
    cdg, c)

@vectorize('float64(float64, float64)')
def abs2(r, i):
    return r**2 + i**2


def projector(ispin):
    if ispin==0:
        return project_exact_up
    return project_exact_dw


def project_add(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace):
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
    n = A.shape[1]
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
    A2 = A**2
    pos_cdg = np.where(A[lead_extract]>1e-18)[0]
    pos_c = np.where(A[lead_inject]>1e-18)[0]
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,pos_cdg.size))
    v_FJ = np.empty((sctF.eigvals.size,sctJ.eigvals.size,pos_c.size))
    for j, pos_j in enumerate(pos_cdg):
        v_JI[...,j] = A2[lead_extract,pos_j] * sgnI * proj_ispin_j(pos_j, cdg, not_full, sctI, sctJ)
    for i, pos_i in enumerate(pos_c):
        v_FJ[...,i] = A2[lead_inject,pos_i] * sgnJ * proj_ispin_i(pos_i, c, not_empty, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = sctJ.eigvals[None,:]-sctI.eigvals[:,None]
    gf2elist = []
    for f,i in np.ndindex(v_FJ.shape[0],v_JI.shape[1]):
        gf2elist.append(Gf2((nupF,ndwF,f),(nupI,ndwI,i),v_FJ[f],v_JI[:,i].copy(),E[i],dE[f,i]))
    return gf2elist


def project_sub(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace):
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
    n = A.shape[1]
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
    A2 = A**2
    pos_c = np.where(A[lead_inject]>1e-18)[0]
    pos_cdg = np.where(A[lead_extract]>1e-18)[0]
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,pos_c.size))
    v_FJ = np.empty((sctF.eigvals.size,sctJ.eigvals.size,pos_cdg.size))
    for j, pos_j in enumerate(pos_c):
        v_JI[...,j] = A2[lead_inject,pos_j] * sgnI * proj_ispin_j(pos_j, c, not_empty, sctI, sctJ)
    for i, pos_i in enumerate(pos_cdg):
        v_FJ[...,i] = A2[lead_extract,pos_i] * sgnJ * proj_ispin_i(pos_i, cdg, not_full, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = -sctJ.eigvals[None,:]+sctI.eigvals[:,None]
    gf2hlist = []
    for f,i in np.ndindex(v_FJ.shape[0],v_JI.shape[1]):
        gf2hlist.append(Gf2((nupF,ndwF,f),(nupI,ndwI,i),v_FJ[f],v_JI[:,i].copy(),E[i],dE[f,i]))
    return gf2hlist


def project_sector(A, lead_extract, lead_inject, nupI, ndwI, espace, ispin=None):
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
    #  /__                  nn'       E -( E   - E  )            nn'      E - (E   - E)               n   n'    L              R     
    #     nn'                               m+    n'                             n'    m+
    #                                                                                                                           
    n = A.shape[1]
    if ispin is None: 
        ispin = [(0,0),(0,1),(1,0),(1,1)]
    sigmalist = [] #dict.fromkeys(ispin)
    for ispin_i, ispin_j in ispin:
        # Electron and hole green functions.
        try:
            gf2elist = project_add(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace)
        except OutOfHilbertError:
            gf2elist = (None,)
        try:
            gf2hlist = project_sub(A, lead_extract, lead_inject, ispin_j, ispin_i, nupI, ndwI, espace)
        except OutOfHilbertError:
            gf2hlist = (None,)
        sigmalist.extend([Sigma(gf2e,gf2h)
                          for gf2e,gf2h in zip_longest(gf2elist,gf2hlist)])
    return {(sigma.idF,sigma.idI):sigma for sigma in sigmalist}


def build_transition_elements(A, lead_extract, lead_inject, egs, espace):
    """Build projector space. Compute all possible matrix elements connecting 
    the ground state and the vectors reached by the latter (including the matrix
    elements connecting the latter vectors).
    """
    # pspace = defaultdict(lambda : np.ndarray((2,2),dtype=object))
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
    sigmadict = {}
    for ns in np.unique(reached_by_gs,axis=0):
        _ispin = [i for i,ds in zip(ispin,dS) if (ns[0]+ds[0],ns[1]+ds[1]) in reached_by_gs]
        _args = ns[0], ns[1], espace, _ispin
        sigmadict.update(
            project_sector(A, lead_extract, lead_inject, *_args))
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
    def __init__(self, idF, idI, vF, vI, E, dE) -> None:
        self.idF = idF
        self.idI = idI
        self.vF = vF
        self.vI = vI
        self.E = E
        self.dE = dE
        self.dS = (idF[0]-idI[0],idF[1]-idI[1])
        
    def __call__(self, z):
        z = np.atleast_1d(z)
        res = np.einsum('j,j,kj->k',self.vF.sum(-1),self.vI.sum(-1),
                        np.reciprocal(z-self.E[None,:]))
        return res


class Sigma:
    #   |                   |  2
    #   | G  (z) + G    (z) |      
    #   |  e         h      |  
    def __init__(self, gf2e, gf2h) -> None:
        if (gf2e is not None) and (gf2h is not None):
            assert np.allclose(gf2e.idF, gf2h.idF)&np.allclose(gf2e.idI, gf2h.idI), """
                Invalid projections. Initial and final states 
                after N+1 and N-1 projections are not the same."""
        self.gf2e = gf2e
        self.gf2h = gf2h
        self.gf2 = self.gf2e or self.gf2h
        self.mu = None        
        
        if (gf2e is not None) and (gf2h is not None):
            self._call = lambda z :  self.gf2e(z) + self.gf2h(z)
            self._integrate = lambda beta, mu : Gamma2(
                self.gf2e.vF[0].sum()*self.gf2e.vI[0].sum(), #A
                self.gf2h.vF[0].sum()*self.gf2h.vI[0].sum(), #B
                self.gf2e.E[0], # epsA
                self.gf2h.E[0], # epsB
                [mu[0]-self.dE,mu[1]], beta)
        else:
            self._call = lambda z :  self.gf2(z)
            self._integrate = lambda beta, mu : Gamma1(
                self.gf2.vF[0].sum()*self.gf2.vI[0].sum(), # A
                self.gf2.E[0], # epsA
                [mu[0]-self.dE,mu[1]], beta)
            
        self._approximate = lambda beta, mu : self.G(beta, self.dE-mu[0]+mu[1]) * self(0.5*(self.dE+sum(mu)))

    @staticmethod
    def G(beta, z):
        if (beta*abs(z))<1e-18:
            return 1.
        if (beta*z)>1e3:
            return 0.
        return z/(np.exp(beta*z)-1)

    def __getattr__(self, name):
        """Default is to return attribute of gf2e."""
        if name in ['idF','idI','dS','dE']:
            return getattr(self.gf2, name)
        raise AttributeError
    
    def integrate(self, beta, mu, approximate=False):
        #                   __  
        #   _ ll'ss'       |       __ll'ss'    
        #  |          =    |   dE  \        (E)
        #    nn'         __|       /__
        #                              nn'
        #             
        mu = tuple(mu)    
        if self.mu!=mu:
            self.gamma = self._approximate(beta, mu) if approximate else self._integrate(beta, mu)
            self.mu = mu
        return self.gamma

    def __call__(self, z):
        res = self._call(z)
        return abs2(res.real, res.imag)


def build_rate_matrix(sigmadict, beta, mu, approximate=False):
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
    sz, odd = np.divmod(len(sigmadict), 2)
    assert ~odd, """
        Invalid sigma list. Each matrix element must contain 
        its complex conjugate.
    """
    idF = set([idF for idF, _ in sigmadict.keys()])
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    W = np.zeros((sz, sz))
    for i, f in np.ndindex(sz, sz):
        if i != f:
            try:
                gamma = sigmadict[map[f],map[i]].integrate(beta, mu, approximate)
            except:
                continue
            W[i,i] -= gamma
            W[f,i] += gamma
    return W


def build_transition_matrix(sigmadict, beta, mu, approximate=False):
    sz, odd = np.divmod(len(sigmadict), 2)
    assert ~odd, """
        Invalid sigma list. Each matrix element must contain 
        its complex conjugate.
    """
    idF = set([idF for idF, _ in sigmadict.keys()])
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    T = np.empty((sz, sz))
    for i, f in np.ndindex(sz, sz):
        try:
            gamma = sigmadict[map[f],map[i]].integrate(beta, mu, approximate)
        except:
            continue
        T[i,f] = gamma
    return T


def screen_transition_elements(sigmadict, egs, espace, cutoff):
    return {(idF, idI):sigma for (idF, idI),sigma in sigmadict.items() 
            if (abs(sigma.dE)<cutoff)
            and(abs(espace[idF[:2]].eigvals[idF[2]]-egs)<cutoff)
            and(abs(espace[idI[:2]].eigvals[idI[2]]-egs)<cutoff)}