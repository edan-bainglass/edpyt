import numpy as np
from numba import vectorize
from itertools import product
from collections import defaultdict
from operator import attrgetter

from edpyt.integrals import Gamma1, Gamma2
from edpyt.lookup import binsearch
from edpyt.operators import cdg, c
from edpyt.sector import OutOfHilbertError

OutOfHilbertError = KeyError

@vectorize('float64(float64, float64)')
def abs2(r, i):
    return r**2 + i**2


def project(spin, position, operator, n, sctI, sctJ):
    """Project sector sctI onto eigenbasis of sector sctJ.

    """
    #                     ____
    #         (+)         \             *             (+)       
    # < j | c     | i > =  \           a   a   < s | c      | s  >    
    #          p           /             j   i    j    p       i  
    #                     /____  
    #                           i,j
    v_JI = np.zeros((sctJ.eigvals.size,sctI.eigvals.size))
    for i in range(sctI.states.size):
        s = sctI.states[i]
        try:
            sgn, f = operator(s, position+spin*n, 2*n)
        except:
            continue
        j = binsearch(sctJ.states, f)
        v_JI += sgn * np.outer(sctJ.eigvecs[j,:],sctI.eigvecs[i,:])
    return v_JI


def excite(spins, particle, n, N, espace):
    """Cotunneling rate for a lead electron to go from lead l' to lead l and
    the central region to go from state n' to state n, i.e.:
     
            l' -> l
            n' -> n     
    """
    #                                   
    #   +/-               -/+               +/-
    #  y  (p,r) =   < n| c    | m  > < m | c      | n' >
    #   nn'                p,s               r,s'
    #                           
    if particle == 'electron':
        operators = [c, cdg]
        sctJ = espace[(N+1,)]
        sign = +1.
    elif particle == 'hole':
        operators = [cdg, c]
        sctJ = espace[(N-1,)]
        sign = -1.
    else:
        raise ValueError
    sctI = sctF = espace[(N,)]
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    v_FJ = np.empty((sctI.eigvals.size,sctJ.eigvals.size,n))
    for j in range(n):
        v_JI[...,j] = project(spins[1], j, operators[1], n, sctI, sctJ)
    for i in range(n):
        v_FJ[...,i] = project(spins[0], i, operators[0], n, sctJ, sctF)
    dE = sctF.eigvals[:,None] - sctI.eigvals[None,:]
    E =  sign * (sctJ.eigvals[None,:] - sctI.eigvals[:,None])
    gfdict = dict()
    for f, i in np.ndindex(v_FJ.shape[0], v_JI.shape[1]):
        gfdict[(N,f),(N,i)] = Gf2(v_FJ[f], v_JI[:,i].copy(), E[i], dE[f,i])
    return gfdict


def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
    """Cotunneling rate from state n to state n' within sector with N electrons.

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
    if N is None:
        assert egs is not None, "Must provide either groud state or particle sector number."
        egs = np.inf
        for qns, sct in espace.items():
            if egs > sct.eigvals.min():
                N = qns[0]
                egs = sct.eigvals.min()
    sigmadict = defaultdict(list) #dict.fromkeys(ispin)
    for spins in product(range(2), repeat=2):
        # Electron and hole green functions.
        try:
            gf2edict = excite(spins, 'electron', n, N, espace)
        except OutOfHilbertError:
            gf2edict = dict() # empty
        try:
            gf2hdict = excite(list(reversed(spins)), 'hole', n, N, espace)
        except OutOfHilbertError:
            gf2hdict = dict()
        # Sigma        
        for idF, idI in set(gf2edict).union(gf2hdict):
            gf2e = gf2edict.pop((idF,idI),None)
            gf2h = gf2hdict.pop((idF,idI),None)
            sigmadict[idF,idI].append(Sigma(gf2e,gf2h))
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
    """
                           
    #    +(-)              1           
    #  y          ---------------- 
    #    nn'       E -( E   - E  )
    #                    m+    n' 
    def __init__(self, vF, vI, E, dE) -> None:
        self.vF = vF
        self.vI = vI
        self.E = E
        self.dE = dE
        
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
    #  |        | 2
    #  | G  (z) |   
    #  |  e     | 
    
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
    #  |        | 2
    #  | G  (z) |   
    #  |  h     | 
    
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
    

# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.74.205438
def build_rate_matrix(sigmadict, beta, mu, A, approx_integral=False):
    #          ___
    #         |     __                      
    #         |    \     _             _   
    #         |  -      |             |            --  
    #         |    /__     k0           01
    #         |        k!=0           __         
    #  W  =   |        _             \     _       --  
    #         |       |            -      |          
    #         |         10           /__     k1  
    #         |                          k!=1
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
            and(abs(espace[idF[:1]].eigvals[idF[1]]-egs)<cutoff)
            and(abs(espace[idI[:1]].eigvals[idI[1]]-egs)<cutoff)}