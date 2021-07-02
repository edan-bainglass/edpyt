import numpy as np
from numba import vectorize, prange

from edpyt.espace import (
    build_espace
)

from edpyt.lookup import (
    get_spin_indices,
    get_state_index,
    binsearch
)

from edpyt.shared import (
    unsiged_dt
)

from edpyt.operators import (
    cdg, check_full
)


# def Gf(q, l):
#     """Green's function kernel wrapper.

#     """
#     # @vectorize('complex64(float64,float64)',nopython=True)
#     def inner(e, eta):
#         res=0.+0.j
#         for i in prange(q.size):
#             res += q[i] / ( e + 1.j*eta - l[i] )
#         return res / Z
#     return inner

class Gf:
    def __init__(self, q, l, Z):
        self.q = np.asarray(q)
        self.l = np.asarray(l)
        self.Z = Z

    def __call__(self, e, eta):
        z = np.atleast_1d(e + 1.j*eta)
        res = np.dot(np.reciprocal(z[:,None]-self.l[None,:]),self.q)
        # res = np.einsum('i,ki',self.q,np.reciprocal(z[:,None]-self.l[None,:]),optimize=True)
        return res / self.Z


def project_exact_up(pos, op, check_occupation, sctI, sctJ):
    """Project states of sector sctI onto eigenbasis of sector sctJ.

    """
    #                      ____
    #          +          \
    # < N'j| c  | Nk > =   \        a     a     , \__/ i',i  | N'i' >  =  op  | Ni >
    #          0           /         i',j  i,k     \/                       0
    #                     /____ i'i
    #                           (lattice sites)
    v0 = np.zeros((sctJ.eigvals.size,sctI.eigvals.size))
    for iupI in range(sctI.dup):
        supI = sctI.states.up[iupI]
        # Check for empty impurity
        if check_occupation(supI, pos): continue
        sgnJ, supJ = op(supI, pos)
        iupJ = binsearch(sctJ.states.up, supJ)
        v0 += np.float64(sgnJ)*sctJ.eigvecs[iupJ::sctJ.dup,:].T.dot(
                               sctI.eigvecs[iupI::sctI.dup,:])
    return v0


def project_exact_dw(pos, op, check_occupation, sctI, sctJ):
    """Project states of sector sctI onto eigenbasis of sector sctJ.

    """
    #                      ____
    #          +          \
    # < N'j| c  | Nk > =   \        a     a     , \__/ i',i  | N'i' >  =  op  | Ni >
    #          0           /         i',j  i,k     \/                       0
    #                     /____ i'i
    #                           (lattice sites)
    v0 = np.zeros((sctJ.eigvals.size,sctI.eigvals.size))
    for idwI in range(sctI.dwn):
        sdwI = sctI.states.dw[idwI]
        # Check for empty impurity
        if check_occupation(sdwI, pos): continue
        sgnJ, sdwJ = op(sdwI, pos)
        idwJ = binsearch(sctJ.states.dw, sdwJ)
        v0 += np.float64(sgnJ)*sctJ.eigvecs[idwJ*sctJ.dup:(idwJ+1)*sctJ.dup,:].T.dot(
                               sctI.eigvecs[idwI*sctI.dup:(idwI+1)*sctI.dup,:])
    return v0


def build_gf_exact(H, V, espace, beta, egs=0., pos=0, ispin=0):
    """Build Green's function with exact diagonalization.

    """
    #
    #             ______
    #             \             (-beta E(N')l')  (-beta E(N)l)
    #         1    \           e                e                           +         2
    # G  =   ---           ----------------------------------   | < N'l'| c  | Nl > |
    # ii      Z    /            iw   - ( E(N')l' - E(N)l )                  i
    #             /_____
    #            N,N',l,l'.
    #
    n = H.shape[-1]
    project_exact = [project_exact_up, project_exact_dw][ispin]
    # espace, egs = build_espace(H, V)
    lambdas = []
    qs = []
    #  ____
    # \
    #  \
    #  /
    # /____ NN'
    for nupI, ndwI in espace.keys():
        sctI = espace[(nupI,ndwI)]
        # N+1 (one more up spin)
        nupJ = nupI+1
        ndwJ = ndwI
        # Cannot have more spin than spin states
        if nupJ > n: continue
        # Arrival sector
        sctJ = espace[(nupJ,ndwJ)]
        #  ____                          ____ 
        # \                  +          \                               +
        #  \        < N'l'| c  | Nl > =  \          c     c    < N'm'| c  | Nn >
        #  /                 0           /           m,l'  n,l          0
        # /____ ll'                     /____ ll'mn
        EI = (sctI.eigvals-egs)[None,:]
        EJ = (sctJ.eigvals-egs)[:,None]
        exponents = np.exp(-beta*EJ) + np.exp(-beta*EI)
        bJ = project_exact(pos, cdg, check_full, sctI, sctJ)
        lambdas.extend((EJ - EI).flatten())
        qs.extend((bJ**2 * exponents).flatten())

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-egs)).sum() for
        (nup, ndw), sct in espace.items())

    # qs = np.array(qs)#/Z
    # lambdas = np.array(lambdas)
    gf = Gf(qs, lambdas, Z)

    return gf#Gf(qs, lambdas)
