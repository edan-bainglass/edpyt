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

    Args:
        spin : s
        position : i
        operato : c or cdg
    """
    #                     ____
    #        +/-          \             *             +/-       
    # < k | c     | l > =  \           a   a   < s | c      | s  >    
    #         i,s           /            k   k    k    i,s     l  
    #                     /____  
    #                           k,l
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


def excite_states(spins, operators, n, sctI, sctJ):
    """Compute particle (electron or hole) excitations for sector.
    
    Args:
        spins : [s', s]
        operators : [c+,c-] for hole [c-,c+] for electron
        n : # of sites
    """
    #                                   
    #   +/-                 -/+               +/-
    #  y         =   < n'| c    | m  > < m | c      | n >
    #   s'n'i,snj            s'i               sj
    #    
    sctF = sctI
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    v_FJ = np.empty((sctI.eigvals.size,sctJ.eigvals.size,n))
    for j in range(n):
        v_JI[...,j] = project(spins[1], j, operators[1], n, sctI, sctJ)
    for i in range(n):
        v_FJ[...,i] = project(spins[0], i, operators[0], n, sctJ, sctF)
    return v_FJ, v_JI

def excite_electron(spins, n, N, espace):
    """Compute electron green's function for N electrons sector.
    
    """                   
    #                     
    #    +               +          -                    1             +
    #  G   (E)    =     A (inject) A (extract)     ---------------   y  
    #   n's'i,nsj        i          j                E+ - En' - E      s'n'i,snj 
    #
    sctI = sctF = espace[(N,)]
    sctJ = espace[(N+1,)]
    v_FJ, v_JI = excite_states(spins, [c, cdg], n, sctI, sctJ)
    dE_FI = sctF.eigvals[:,None] - sctI.eigvals[None,:]
    dE_FJ = sctJ.eigvals[None,:] - sctF.eigvals[:,None]
    gfdict = dict()
    for f, i in np.ndindex(v_FJ.shape[0], v_JI.shape[1]):
        gfdict[(N,f),(N,i)] = Gf2(v_FJ[f], v_JI[:,i].copy(), dE_FJ[f], dE_FI[f,i])
    return gfdict


def excite_hole(spins, n, N, espace):
    """Compute hole green's function for N electrons sector.
    
    """
    #                     
    #    -              +           -                    1             -
    #  G   (E)    =    A (extract) A (inject)     ---------------   y  
    #   n's'i,nsj        i          j                En - E- - E      s'n'i,snj
    #
    sctI = sctF = espace[(N,)]
    sctJ = espace[(N-1,)]
    v_FJ, v_JI = excite_states(spins, [cdg, c], n, sctI, sctJ)
    dE_FI = sctF.eigvals[:,None] - sctI.eigvals[None,:]
    dE_IJ = sctI.eigvals[:,None] - sctJ.eigvals[None,:]
    gfdict = dict()
    for f, i in np.ndindex(v_FJ.shape[0], v_JI.shape[1]):
        gfdict[(N,f),(N,i)] = Gf2(v_FJ[f], v_JI[:,i].copy(), dE_IJ[i], dE_FI[f,i])
    return gfdict


def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
    """Cotunneling rate from state n to state n' within sector with N electrons.

    NOTE: that spins are interchanged to ensure equal initial and final states. 
    """
    #                |     +                 -          |
    #  S  (E)      = |   G   (E)        +  G    (E)     | . .  n  (E - En' - En - mu(extract))  ( 1 - n  (E - mu(inject)))
    #   s'n'i,snj    |     n's'i,nsj         n'si,ns'j  |       F                                       F                 
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
            gf2edict = excite_electron(spins, n, N, espace)
        except OutOfHilbertError:
            gf2edict = dict() # empty
        try:
            gf2hdict = excite_hole(list(reversed(spins)), n, N, espace)
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
                        np.reciprocal(self.E[None,:]-z[:,None]))
        return res


nF = lambda x: 1/(np.exp(x)+1)
nF.__doc__ = "Fermi function."

G = lambda x: x/(np.exp(x)-1)
G.__doc__ = "Fermi function."

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
               ) * self(0.5*(self.dE-mu[extract]+mu[inject]), A, extract, inject)
    
    def numeric(self, A, extract, inject, beta, mu):
        mu_extract = mu[extract] - self.dE
        mu_inject = mu[inject]
        # if mu_extract - mu_inject < 0.:
        #     return 0.
        mu_low, mu_high = sorted((mu_extract, mu_inject))
        eners = np.linspace(mu_low-4/beta, mu_high+4/beta, 200, endpoint=True)
        return np.trapz(self(eners,A,extract,inject)
                        * nF(beta*(eners-mu_extract)
                        * nF(-beta*(eners-mu_inject))), # (1 - nF)
                        eners)

class _Sigmae(_Sigma):
    """Electron."""
    #  |        | 2
    #  | G  (z) |   
    #  |  e     | 
    
    def integrate(self, A, extract, inject, beta, mu):
        a = A[inject].dot(self.gf2.vF.T)*A[extract].dot(self.gf2.vI.T)
        nnz_a, = np.nonzero(a)
        if nnz_a.any():
            j = nnz_a.min()
            return Gamma1(
                a[j], # A
                -self.gf2.E[j], # epsA
                [mu[extract]-self.dE,mu[inject]], beta)
        else:
            return 0.

    def __call__(self, z, A, extract, inject):
        res = self.gf2(z, A[inject], A[extract])
        return abs2(res.real, res.imag)
    

class _Sigmah(_Sigma):
    """Hole."""
    #  |        | 2
    #  | G  (z) |   
    #  |  h     | 
    
    def integrate(self, A, extract, inject, beta, mu):
        a = A[extract].dot(self.gf2.vF.T)*A[inject].dot(self.gf2.vI.T)
        nnz_a, = np.nonzero(a)
        if nnz_a.any():
            j = nnz_a.min()
            return Gamma1(
                a[j], # A
                -self.gf2.E[j], # epsA
                [mu[extract]-self.dE,mu[inject]], beta)
        else:
            return 0.
    
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
        a = A[inject].dot(self.gf2e.vF.T)*A[extract].dot(self.gf2e.vI.T)
        b = A[extract].dot(self.gf2h.vF.T)*A[inject].dot(self.gf2h.vI.T)
        nnz_a, = np.nonzero(a)
        nnz_b, = np.nonzero(b)
        if nnz_a.any():
            j = nnz_a.min()
            if nnz_b.any():
                k = nnz_b.min()
                return Gamma2(
                    a[j], #A
                    b[k], #B
                    -self.gf2e.E[j], # epsA
                    -self.gf2h.E[k], # epsB
                    [mu[extract]-self.dE,mu[inject]], beta)
            else:
                return Gamma1(
                    a[j], # A
                    -self.gf2e.E[j], # epsA
                    [mu[extract]-self.dE,mu[inject]], beta)
        elif nnz_b.any():
            k = nnz_b.min()
            return Gamma1(
              b[k],
              -self.gf2h.E[k],
              [mu[extract]-self.dE,mu[inject]], beta)
        else:
            return 0.

    def __call__(self, z, A, extract, inject):
        res = self.gf2e(z, A[inject], A[extract]) + self.gf2h(z, A[extract], A[inject])
        return abs2(res.real, res.imag)
    

# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.74.205438
def build_rate_matrix(sigmadict, beta, mu, A, integrate_method='approximate'):
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
    integrate = attrgetter(integrate_method)
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
                except KeyError:
                    continue
                W[i,i] -= gamma
                W[f,i] += gamma
    return W


def build_transition_matrix(sigmadict, beta, mu, A, extract, inject, integrate_method='approximate'):
    sz, odd = np.divmod(len(sigmadict), 2)
    assert ~odd, """
        Invalid sigma list. Each matrix element must contain 
        its complex conjugate.
    """
    integrate = attrgetter(integrate_method)
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
        except KeyError:
            continue
        T[f,i] = gamma
    return T


def screen_transition_elements(sigmadict, egs, espace, cutoff):
    return {(idF, idI):sigma for (idF, idI),sigma in sigmadict.items() 
            if (abs(sigma[0].dE)<cutoff)
            and(abs(espace[idF[:1]].eigvals[idF[1]]-egs)<cutoff)
            and(abs(espace[idI[:1]].eigvals[idI[1]]-egs)<cutoff)}