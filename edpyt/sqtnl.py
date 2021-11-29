from collections import defaultdict
import numpy as np

from edpyt.cotnl import (c, cdg,
    Gf2e, Gf2h, check_transition_elements, dict_to_matrix, nF, project)#,
    # build_transition_elements as build_cotunneling_elements)


# def extract_sequential_from_cotunneling(sigmadict):
#     gfdict = defaultdict(list)
#     for (idF,idI), sigmalist in sigmadict.items():
#         N, i = idI
#         N, f = idF
#         if i != f:
#             continue
#         for sigma in sigmalist:
#             for gf in sigma.gf2:
#                 if isinstance(gf, Gf2e):
#                     for j in range(gf.vI.shape[0]):
#                         gfdict[(N+1,j),(N,i)].append(Gfe(gf.vI[j], gf.E[j]))#+gf.dE)
#                         gfdict[(N,f),(N+1,j)].append(Gfh(gf.vF[j], gf.E[j]))
#                 elif isinstance(gf, Gf2h):
#                     for j in range(gf.vI.shape[0]):
#                         gfdict[(N-1,j),(N,i)].append(Gfh(gf.vI[j], gf.E[j]))
#                         gfdict[(N,f),(N-1,j)].append(Gfe(gf.vF[j], gf.E[j]))#+gf.dE)
#     return gfdict


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
        gfdict[(N+1,j),(N,i)] = Gfe(v_JI[j,i], dE_JI[j,i])
    return gfdict


def excite_hole(spin, n, N, espace):
    sctI = espace[(N,)]
    sctJ = espace[(N-1,)]
    v_JI = excite_states(spin, c, n, sctI, sctJ)
    dE_JI = - sctJ.eigvals[:,None] + sctI.eigvals[None,:]
    gfdict = dict()
    for j, i in np.ndindex(dE_JI.shape):
        gfdict[(N-1,j),(N,i)] = Gfh(v_JI[j,i], dE_JI[j,i])
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
        # return screen_transition_elements(sigmadict, egs, espace, cutoff)
    return sigmadict

# def build_transition_elements(n, espace, N=None, egs=None, cutoff=None):
#     sigmadict = build_cotunneling_elements(n, espace, N, egs, cutoff)
#     return extract_sequential_from_cotunneling(sigmadict)


class Gf:
    """Green's function.
    
    Args:
        E : (E+ - En') or (En' - E-)
        v : <m+|c+|n> or <m-|c|n>
    """
      
    def __init__(self, v, E) -> None:
        self.v = v
        self.E = E

    def y(self, a):
        return np.inner(self.v, a)**2


class Gfe(Gf):
    """Electron.
        |<N+1|c+|N>|^2 nF(E-mu)
    """
    def __call__(self, a, beta, mu):
        return self.y(a)*nF(beta*(self.E-mu))


class Gfh(Gf):
    """Hole.
        |<m-|c+|n'>|^2 (1-nF(E-mu))
    NOTE: nF(-E)==(1-nF(E))
    """
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
            T[idF,idI] += gamma[extract]
            T[idF,idI] -= gamma[inject]
        else:
            T[idF,idI] += gamma[inject]
            T[idF,idI] -= gamma[extract]
    if build_matrices:
        return map(lambda D: dict_to_matrix(D), [W,T])
    return W, T