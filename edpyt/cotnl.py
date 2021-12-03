from functools import lru_cache
from numba.core.decorators import njit
import numpy as np
from numba import vectorize
from itertools import product
from collections import defaultdict
from operator import attrgetter

from numpy.lib.function_base import extract
from traitlets.traitlets import default

from edpyt.integrals import I1, I2, Gamma1, Gamma2, Gamma4
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
        gfdict[(N,f),(N,i)] = Gf2e(v_FJ[f], v_JI[:,i].copy(), dE_FJ[f], dE_FI[f,i])
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
        gfdict[(N,f),(N,i)] = Gf2h(v_FJ[f], v_JI[:,i].copy(), dE_IJ[i], dE_FI[f,i])
    return gfdict


def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
    """Cotunneling rate from state n to state n' within sector with N electrons.

    NOTE: that spins are interchanged to ensure equal initial and final states. 
    """
    #                |     +                 -          |
    #  S  (E)      = |   G   (E)        +  G    (E)     | . n  (E - En' - En - mu(extract))  ( 1 - n  (E - mu(inject)))
    #   s'n'i,snj    |     n's'i,nsj         n'si,ns'j  |    F                                       F                 
    if N is None:
        assert egs is not None, "Must provide either groud state or particle sector number."
        egs = np.inf
        for qns, sct in espace.items():
            if egs > sct.eigvals.min():
                N = qns[0]
                egs = sct.eigvals.min()
    sigmadict = defaultdict(list) #dict.fromkeys(ispin)
    N_max = 2*n
    for spins in product(range(2), repeat=2):
        # Electron and hole green functions.
        if N+1 < N_max:
            gf2edict = excite_electron(spins, n, N, espace)
        else:
            gf2edict = dict() # empty
        if N-1 > 0:
            gf2hdict = excite_hole(list(reversed(spins)), n, N, espace)
        else:
            gf2hdict = dict()
        # Sigma        
        for idF, idI in set(gf2edict).union(gf2hdict):
            gf2e = gf2edict.pop((idF,idI),None)
            gf2h = gf2hdict.pop((idF,idI),None)
            sigmadict[idF,idI].append(Sigma(spins,gf2e,gf2h))
    sigmadict = {(idF,idI):sigmadict[(idF,idI)] for (idF,idI) in sorted(sigmadict.keys())}
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
    
    def y(self, A, extract, inject):
        raise NotImplementedError
    
    def __call__(self, z, A, extract, inject):
        z = np.atleast_1d(z)
        res = np.einsum('j,kj->k',self.y(A, extract, inject),
                        np.reciprocal(self.E[None,:]-z[:,None]))
        return res


class Gf2e(Gf2):
    
    def __init__(self, vF, vI, E, dE) -> None:
        super().__init__(vF, vI, E, dE)
        
    def y(self, A, extract, inject):
        # return np.sum((A[inject]*self.vF)[:,:,None]
        #             * (A[extract]*self.vI)[:,None,:],(1,2))
        return A[inject].dot(self.vF.T)*A[extract].dot(self.vI.T)


class Gf2h(Gf2):
    
    def __init__(self, vF, vI, E, dE) -> None:
        super().__init__(vF, vI, E, dE)
        
    def y(self, A, extract, inject):
        # return np.sum((A[extract]*self.vF)[:,:,None]
        #             * (A[inject]*self.vI)[:,None,:],(1,2))
        return A[extract].dot(self.vF.T)*A[inject].dot(self.vI.T)        


# def nF(x):
#     """Fermi distribution."""
#     if x>1e3:
#         return 0.
#     return 1/(np.exp(x)+1)
nF = lambda x: 1/(np.exp(x)+1)
nF.__doc__ = "Fermi function."


def G(beta, z):
    """Helper function for approximate solution."""
    if (beta*abs(z))<1e-18:
        return 1.
    if (beta*z)>1e3:
        return 0.
    return z/(np.exp(beta*z)-1)

# from numba import vectorize, guvectorize

# @guvectorize('(float64,float64[:],float64[:])','(),(n)->(n)')
# def G(a, x, out):
#     for i in range(x.size):
#         if -1e-4 < x[i] < 1e-4:
#             out[i] = 1/a - x[i]/2 + a*x[i]**2/12 - a**3*x[i]**4/720
#         elif x[i]>1e13:
#             out[i] = 0.
#         else:
#             out[i] = x[i]/(np.exp(a*x[i])-1)

class Sigma:
    
    def __init__(self, spins, gf2e=None, gf2h=None) -> None:
        self.spins = spins
        self.gf2 = [gf for gf in [gf2e,gf2h] if gf is not None]
        self.dE = self.gf2[0].dE

    def __repr__(self) -> str:
        return f'Sigma(spins={self.spins},dE={self.dE:.5E})'

    def __call__(self, z, A, extract, inject):
        res = sum(gf2(z, A, extract, inject) for gf2 in self.gf2)
        return res**2
    
    def approximate(self, A, extract, inject, beta, mu):
        x = self.dE-mu[extract]+mu[inject]
        # return G(beta, x) * self(0.5*x, A, extract, inject)
        return G(beta, x) * self(0.5*(self.dE+sum(mu)), A, extract, inject)
    
    def exact(self, A, extract, inject, beta, mu):
        mu = [mu[extract]-self.dE,mu[inject]]
        Y = [Y for gf in self.gf2 for Y in gf.y(A, extract, inject)]
        E = [-E for gf in self.gf2 for E in gf.E]
        if len(Y) == 1:
            return Gamma1(*Y,*E,mu,beta)
        elif len(Y) == 2:
            return Gamma2(*Y,*E,mu,beta)
        elif len(Y) == 4:
            return Gamma4(*Y,*E,mu,beta)
        else:
            raise NotImplementedError("Integral with {len(Y)} terms not yet implemented.")

    def integrate(self, A, extract, inject, beta, mu):
        mu_extract = mu[extract] - self.dE
        mu_inject = mu[inject]
        mu_low, mu_high = sorted((mu_extract, mu_inject))
        eners = np.linspace(mu_low-4/beta, mu_high+4/beta, 200, endpoint=True)
        return np.trapz(self(eners,A,extract,inject)
                        * nF(beta*(eners-mu_extract)
                        * nF(-beta*(eners-mu_inject))), # (1 - nF)
                        eners)


def get_unique_ids(sigmadict):
    sorted_keys = lambda keys, idx: sorted(set(map(lambda ids: ids[idx], keys)))
    idF = sorted_keys(sigmadict.keys(), 0)
    idI = sorted_keys(sigmadict.keys(), 1)
    return set(idF), set(idI)


def check_transition_elements(sigmadict):
    idF, idI = get_unique_ids(sigmadict)    
    assert idF.union(idI) == idF, "Missing transition rates. Must provide n->n' & n'->n."


# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.74.205438
def build_rate_and_transition_matrices(sigmadict, beta, mu, A, extract, inject, 
                                       integrate_method='approximate', build_matrices=True):
    #                ___                                  ___     _   _
    #  |     |      |     __                                 |   |     |     
    #  |     |      |    \     _             _               |   |     | 
    #  | |0> |      |  -      |             |            --  |   |  P  |     
    #  |     |      |    /__     k0           01             |   |   0 | 
    #  |     |      |        k!=0           __               |   |     | 
    #  |     |  =   |        _             \     _       --  |   |     |      
    #  | |1> |      |       |            -      |            |   |  P  |      
    #  |     |      |         10           /__     k1        |   |   1 | 
    #  |     |      |                          k!=1          |   |     |     
    #  |  :  |      |       :                :          \    |   |  :  |     
    check_transition_elements(sigmadict)
    integrate = attrgetter(integrate_method)
    W = defaultdict(lambda: 0.)
    T = defaultdict(lambda: 0.)
    for (idF,idI), sigmalist in sigmadict.items():
        gamma = np.zeros((2,2))
        for lead1, lead2 in np.ndindex(2, 2):
            for sigma in sigmalist:
                gamma[lead1,lead2] += integrate(sigma)(A, lead1, lead2, beta, mu)
        if idF != idI:
        # if idF[1] != idI[1]:
            W[idI,idI] -= gamma.sum()
            W[idF,idI] += gamma.sum()
        T[idF,idI] += gamma[extract,inject] - gamma[inject,extract]
    if build_matrices:
        return map(lambda D: dict_to_matrix(D), [W,T])
    return W, T


def dict_to_matrix(D):
    id1, id2 = get_unique_ids(D)
    iD = sorted(id1.union(id2))
    map = {id:i for i,id in enumerate(iD)}
    sz = len(iD)
    M = np.zeros((sz,sz))
    for (id1,id2), val in D.items():
        M[map[id1],map[id2]] += val
    return M


def screen_transition_elements(sigmadict, egs, espace, cutoff):
    sigmadict = {(idF,idI):sigmalist for (idF,idI),sigmalist in sigmadict.items()
                if abs(sigmalist[0].dE)<cutoff
                and(abs(espace[idF[:1]].eigvals[idF[1]]-egs)<cutoff)
                and(abs(espace[idI[:1]].eigvals[idI[1]]-egs)<cutoff)}
    # return sigmadict
    n = sigmadict[next(iter(sigmadict))][0].gf2[0].vF.shape[1]
    A = np.ones((2,n))
    screened = {}
    for (idF,idI), sigmalist in sigmadict.items():
        sigmas = []
        for sigma in sigmalist:
            gfs = []
            for gf in sigma.gf2:
                y = gf.y(A, 1, 0)
                nnz_idx = np.abs(y) > 1e-10
                if any(nnz_idx):
                    gfs.append(gf)
                    gf.vF = gf.vF[nnz_idx,:]
                    gf.vI = gf.vI[nnz_idx,:]
                    gf.E = gf.E[nnz_idx]
            sigma.gf2 = gfs
            if gfs: # list not empty
                sigmas.append(sigma)
        if sigmas: # list not empty
            screened[(idF,idI)] = sigmas
    return screened