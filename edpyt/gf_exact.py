import numpy as np
from numba import vectorize, prange

from espace import (
    build_espace
)

from lookup import (
    get_spin_indices,
    get_state_index
)

from shared import (
    unsiged_dt
)

from operators import (
    cdg
)


def Gf(q, l):
    """Green's function kernel wrapper.

    """
    @vectorize('complex64(float64,float64)',nopython=True)
    def inner(e, eta):
        res=0.+0.j
        for i in prange(q.size):
            res += q[i] / ( e + 1.j*eta - l[i] )
        return res
    return inner


def build_gf_exact(H, V, beta, mu=0.):
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
    n = H.shape[0]
    espace, egs = build_espace(H, V)
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
        #  ____
        # \
        #  \
        #  /
        # /____ ll'
        for iI, iJ in np.ndindex(sctI.d, sctJ.d):
            residual = 0.
            EI = sctI.eigvals[iI] - mu*(nupI+ndwI)
            EJ = sctJ.eigvals[iJ] - mu*(nupJ+ndwJ)
            exponent = np.exp(-beta*EI)+np.exp(-beta*EJ)
            # if exponent<1e-9: continue
            #                      ____
            #          +          \                             +
            # < N'l'| c  | Nl > =  \        c     c    < N'm'| c  | Nn >
            #          0           /         m,l'  n,l          0
            #                     /____ mn
            for iL in range(sctI.d):
                iupI, idwI = get_spin_indices(iL, sctI.dup, sctI.dwn)
                supI = sctI.states.up[iupI]
                sdwJ = sctI.states.dw[idwI]
                # If not empty (imputiry)
                if supI&unsiged_dt(1): continue
                sgnJ, supJ = cdg(supI, 0)
                iupJ = np.searchsorted(sctJ.states.up, supJ)
                idwJ = idwI
                iM = get_state_index(iupJ, idwJ, sctJ.dup)
                residual += sctJ.eigvecs[iM,iJ]*sgnJ*sctI.eigvecs[iL,iI]
            lambdas.append(EJ - EI)
            qs.append(residual**2 * exponent)

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-mu*(nup+ndw))).sum() for
        (nup, ndw), sct in espace.items())

    qs = np.array(qs)/Z
    lambdas = np.array(lambdas)

    return Gf(qs, lambdas)
