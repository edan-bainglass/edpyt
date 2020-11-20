import numpy as np
from numba import njit, jit
from multiprocessing import Pool

from lookup import (
    get_spin_indices,
    get_state_index,
    binsearch
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
    # @njit
    def inner(e, eta, n=sz):
        if n==1:
            return b[sz-n] / (e + 1.j*eta - a[sz-n])
        else:
            return b[sz-n] / (e + 1.j*eta - a[sz-n] - inner(e, eta, n-1))
    inner.a = a
    inner.b = b
    return inner


class Gf:
    def __init__(self):
        self.funcs = []
        self.Z = np.inf
    def add(self, a, b):
        self.funcs.append(
            continued_fraction(a, b)
        )
    def __call__(self, e, eta):
        out = np.sum([f(e, eta) for f in self.funcs], axis=0)
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
                sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
                v0 = np.zeros(sctJ.d)
                #             +
                # < (N+1)l'| c(i)  | Nl >
                #
                idwI = np.arange(sctI.dwn) * sctI.dup
                idwJ = np.arange(sctJ.dwn) * sctJ.dup
                for iupI in range(sctI.dup):
                    supI = sctI.states.up[iupI]
                    # Check for empty impurity
                    if supI&unsiged_dt(1): continue
                    sgnJ, supJ = cdg(supI, 0)
                    iupJ = binsearch(sctJ.states.up, supJ)
                    iL = iupI + idwI
                    iM = iupJ + idwJ
                    v0[iM] = sgnJ*sctI.eigvecs[iL,iI]

                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                bJ **= 2
                bJ[0] *= exponent
                gf.add((aJ-sctI.eigvals[iI]), bJ)
            # N-1 (one more up spin)
            nupJ = nupI-1
            ndwJ = ndwI
            # Cannot have negative spin
            if nupJ >= 0:
                # Arrival sector
                sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
                v0 = np.zeros(sctJ.d)
                #
                # < (N-1)l'| c(i)  | Nl >
                #
                idwI = np.arange(sctI.dwn) * sctI.dup
                idwJ = np.arange(sctJ.dwn) * sctJ.dup
                for iupI in range(sctI.dup):
                    supI = sctI.states.up[iupI]
                    # Check for occupied impurity
                    if not supI&unsiged_dt(1): continue
                    sgnJ, supJ = c(supI, 0)
                    iupJ = binsearch(sctJ.states.up, supJ)
                    iL = iupI + idwI
                    iM = iupJ + idwJ
                    v0[iM] = sgnJ*sctI.eigvecs[iL,iI]

                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                bJ **= 2
                bJ[0] *= exponent
                gf.add(-(aJ-sctI.eigvals[iI]), bJ)

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-mu*(nup+ndw))).sum() for
        (nup, ndw), sct in espace.items())
    gf.Z = Z

    return gf
