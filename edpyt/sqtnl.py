from collections import defaultdict
from re import escape
import numpy as np

from edpyt.cotnl import (c, cdg,
    Gf2e, Gf2h, check_transition_elements, dict_to_matrix, project)#,
    # build_transition_elements as build_cotunneling_elements)


def extract_sequential_from_cotunneling(sigmadict):
    gfdict = defaultdict(list)
    gfedict = defaultdict(list)
    gfhdict = defaultdict(list)
    for (idF,idI), sigmalist in sigmadict.items():
        N, i = idI
        N, f = idF
        if i != f:
            continue
        for sigma in sigmalist:
            spins = sigma.spins
            for gf in sigma.gf2:
                if isinstance(gf, Gf2e):
                    for j in range(gf.vI.shape[0]):
                        gfedict[(N+1,j),(N,i),spins[1]].append(Gfe(spins[1],gf.vI[j],gf.E[j]))#+gf.dE)
                        gfhdict[(N,f),(N+1,j),spins[0]].append(Gfh(spins[0],gf.vF[j],gf.E[j]))
                elif isinstance(gf, Gf2h):
                    for j in range(gf.vI.shape[0]):
                        gfhdict[(N-1,j),(N,i),spins[1]].append(Gfh(spins[1],gf.vI[j],gf.E[j]))
                        gfedict[(N,f),(N-1,j),spins[0]].append(Gfe(spins[0],gf.vF[j],gf.E[j]))#+gf.dE)
    for (idF,idI,spin),gflist in gfedict.items():
        gfdict[idF,idI].extend(gflist)
    for (idF,idI,spin),gflist in gfhdict.items():
        gfdict[idF,idI].extend(gflist)
    return gfdict


def excite_states(spin, operator, n, sctI, sctJ):
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    for j in range(n):
        v_JI[...,j] = project(spin, j, operator, n, sctI, sctJ)
    return v_JI


def excite_electron(spin, n, N, espace):
    sctI = espace[(N,)]
    sctJ = espace[(N+1,)]
    v_JI = excite_states(spin, cdg, n, sctI, sctJ)
    dE_JI = sctJ.eigvals[:,None] - sctI.eigvals[None,:]
    gfdict = dict()
    for j, i in np.ndindex(dE_JI.shape):
        gfdict[(N+1,j),(N,i)] = Gfe(spin,v_JI[j,i],dE_JI[j,i])
    return gfdict


def excite_hole(spin, n, N, espace):
    sctI = espace[(N,)]
    sctJ = espace[(N-1,)]
    v_JI = excite_states(spin, c, n, sctI, sctJ)
    dE_JI = - sctJ.eigvals[:,None] + sctI.eigvals[None,:]
    gfdict = dict()
    for j, i in np.ndindex(dE_JI.shape):
        gfdict[(N-1,j),(N,i)] = Gfh(spin,v_JI[j,i],dE_JI[j,i])
    return gfdict


def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
    if N is None:
        assert egs is not None, "Must provide either groud state or particle sector number."
        egs = np.inf
        for qns, sct in espace.items():
            if egs > sct.eigvals.min():
                N = qns[0]
                egs = sct.eigvals.min()
    sigmadict = defaultdict(list)
    N_max = 2*n
    for spin in range(2):
        # Electron and hole green functions.
        if N+1 < N_max:
            gfdict = excite_electron(spin, n, N, espace)
            gfdict.update(excite_hole(spin, n, N+1, espace))
            for k in gfdict: sigmadict[k].append(gfdict[k])
        if N-1 > 0:
            gfdict = excite_hole(spin, n, N, espace)
            gfdict.update(excite_electron(spin, n, N-1, espace))
            for k in gfdict: sigmadict[k].append(gfdict[k])
    # if cutoff is not None:
    #     return screen_transition_elements(sigmadict, egs, espace, cutoff)
    sigmadict = {(idF,idI):sigmadict[(idF,idI)] for (idF,idI) in sorted(sigmadict.keys())}
    return sigmadict

# def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
#     sigmadict = build_cotunneling_elements(n, espace, N, egs, cutoff)
#     return extract_sequential_from_cotunneling(sigmadict)


def nF(x):
    if x > 1e2:
        return 0.
    return 1./(np.exp(x)+1.)

class Gf:
    """Green's function.
    
    Args:
        E : (E+ - En') or (En' - E-)
        v : <m+|c+|n> or <m-|c|n>
    """
      
    def __init__(self, spin, v, E) -> None:
        self.spin = spin
        self.v = v
        self.E = E

    def y(self, a):
        return np.inner(self.v, a)**2


class Gfe(Gf):
    """Electron.
        |<N+1|c+|N>|^2 nF(E-mu)
    """

    def __repr__(self) -> str:
        return f'Gfe(spin={self.spin},E={self.E}.'
    
    def __call__(self, a, beta, mu):
        return self.y(a)*nF(beta*(self.E-mu))


class Gfh(Gf):
    """Hole.
        |<m-|c+|n'>|^2 (1-nF(E-mu))
    NOTE: nF(-E)==(1-nF(E))
    """
    
    def __repr__(self) -> str:
        return f'Gfh(spin={self.spin},E={self.E}.'
    
    def __call__(self, a, beta, mu):
        return self.y(a)*nF(-beta*(self.E-mu))


def build_rate_and_transition_matrices(gfdict, beta, mu, A, extract, inject, build_matrices=True):
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
    check_transition_elements(gfdict)
    W = defaultdict(lambda: 0.)
    T = defaultdict(lambda: 0.)
    for (idF,idI), gflist in gfdict.items():
        gamma = np.zeros(2)
        for lead in range(2):
            for gf in gflist:
                gamma[lead] += gf(A[lead], beta, mu[lead])
        if idF != idI:
            W[idI,idI] -= gamma.sum()
            W[idF,idI] += gamma.sum()
        if isinstance(gf, Gfe):
            # T[idF,idI] += gamma[extract]
            T[idF,idI] -= gamma[inject]
        elif isinstance(gf, Gfh):
            T[idF,idI] += gamma[inject]
            # T[idF,idI] -= gamma[extract]
    if build_matrices:
        return map(lambda D: dict_to_matrix(D), [W,T])
    return W, T


# def screen_transition_elements(gfdict, egs, espace, cutoff):
#     gfdict = {(idF,idI):sigmalist for (idF,idI),sigmalist in gfdict.items()
#                if abs(sigmalist[0].dE)<cutoff
#                and(abs(espace[idF[:1]].eigvals[idF[1]]-egs)<cutoff)
#                and(abs(espace[idI[:1]].eigvals[idI[1]]-egs)<cutoff)}
#     n = gfdict[next(iter(gfdict))][0].gf2[0].vF.shape[1]
#     A = np.ones((2,n))
#     screened = {}
#     for (idF,idI), sigmalist in gfdict.items():
#         sigmas = []
#         for sigma in sigmalist:
#             gfs = []
#             for gf in sigma.gf2:
#                 y = gf.y(A, 1, 0)
#                 nnz_idx = np.abs(y) > 1e-10
#                 if any(nnz_idx):
#                     gfs.append(gf)
#                     gf.vF = gf.vF[nnz_idx,:]
#                     gf.vI = gf.vI[nnz_idx,:]
#                     gf.E = gf.E[nnz_idx]
#             sigma.gf2 = gfs
#             if gfs: # list not empty
#                 sigmas.append(sigma)
#         if sigmas: # list not empty
#             screened[(idF,idI)] = sigmas
#     return screened