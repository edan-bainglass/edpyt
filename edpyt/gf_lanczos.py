import numpy as np
from numba import njit
from multiprocessing import Pool

from lookup import (
    get_spin_indices,
    get_state_index
)

from shared import (
    unsiged_dt
)

from operators import (
    cdg,
    c
)

from matvec_product import (
    matvec_operator
)

from build_mb_ham import (
    build_mb_ham
)

from lanczos import (
    build_sl_tridiag
)

from espace import (
    build_empty_sector
)

def continued_fraction(a, b):
    sz = a.size
    # @njit('complex64(float64,float64,int64)')
    def inner(e, eta, n=sz):
        if n==0:
            return 0
        else:
            return b[sz-n] / (e + 1.j*eta - a[sz-n] - inner(e, eta, n-1))
    return inner


class Gf:
    def __init__(self):
        self.funcs = []
        self.Z = np.inf
    def add(self, a, b):
        self.funcs.append(
            continued_fraction(a, b)
        )
    def __call__(self, z):
        pool = Pool(4)
        rets = [pool.appy(f, args=(z,)) for f in self.funcs]
        out = np.sum([r.get() for r in rets], axis=0)
        pool.close()
        return out / self.Z



def build_gf_lanczos(H, V, espace, beta, mu=0.):
    """Build Green's function with exact diagonalization.

    """
    #
    #             ______
    #             \                              _                                                      +               _
    #         1    \            (-beta E(N)l)   |      | < (N-1)l'| c(i)  | Nl > |        | < (N+1)l'| c(i)  | Nl > |    |
    # G  =   ---               e                |    ------------------------------  +  ------------------------------   |
    # ii      Z    /                            |_     iw   - ( E(N)l - E(N-1)l' )        iw   - ( E(N+1)l' - E(N)l )   _|
    #             /_____
    #            N,l,l'.
    #
    n = H.shape[0]
    gf = Gf()
    #  ____
    # \
    #  \
    #  /
    # /____ N
    for nupI, ndwI in espace.keys():
        sctI = espace[(nupI,ndwI)]
        #  ____
        # \
        #  \
        #  /
        # /____ l
        for iI in np.ndindex(sctI.eigvals.size):
            exponent = np.exp(-beta*(sctI.eigvals[iI]-mu*(nupI+ndwI)))
            # N+1 (one more up spin)
            nupJ = nupI+1
            ndwJ = ndwI
            # Cannot have more spin than spin states
            if nupJ <= n:
                # Arrival sector
                sctJ = build_empty_sector(n, nupJ, ndwJ)
                v0 = np.zeros(sctJ.d)
                #             +
                # < (N+1)l'| c(i)  | Nl >
                #
                for iL in range(sctI.d):
                    iupI, idwI = get_spin_indices(iL, sctI.dup, sctI.dwn)
                    supI = sctI.states.up[iupI]
                    sdwJ = sctI.states.dw[idwI]
                    # Check for empty impurity
                    if supI&unsiged_dt(1): continue
                    sgnJ, supJ = cdg(supI, 0)
                    iupJ = np.searchsorted(sctJ.states.up, supJ)
                    idwJ = idwI
                    iM = get_state_index(iupJ, idwJ, sctJ.dup)
                    v0[iM] = sgnJ*sctI.eigvecs[iL,iI]

                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                bJ **= 2
                bJ[0] *= exponent
                gf.add(aJ, bJ)
            # N-1 (one more up spin)
            nupJ = nupI-1
            ndwJ = ndwI
            # Cannot have negative spin
            if nupJ >= 0:
                # Arrival sector
                sctJ = build_empty_sector(n, nupJ, ndwJ)
                v0 = np.zeros(sctJ.d)
                #
                # < (N-1)l'| c(i)  | Nl >
                #
                for iL in range(sctI.d):
                    iupI, idwI = get_spin_indices(iL, sctI.dup, sctI.dwn)
                    supI = sctI.states.up[iupI]
                    sdwJ = sctI.states.dw[idwI]
                    # Check for occupied imputiry
                    if not supI&unsiged_dt(1): continue
                    sgnJ, supJ = c(supI, 0)
                    iupJ = np.searchsorted(sctJ.states.up, supJ)
                    idwJ = idwI
                    iM = get_state_index(iupJ, idwJ, sctJ.dup)
                    v0[iM] = sgnJ*sctI.eigvecs[iL,iI]

                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                bJ **= 2
                bJ[0] *= exponent
                gf.add(aJ, bJ)

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-mu*(nup+ndw))).sum() for
        (nup, ndw), sct in espace.items())
    gf.Z = Z

    return gf
