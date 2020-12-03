import numpy as np
from numba import njit, prange, vectorize
from multiprocessing import Pool

from edpyt.lookup import (
    get_spin_indices,
    get_state_index,
    binsearch
)

from edpyt.shared import (
    unsiged_dt
)

from edpyt.operators import (
    cdg,
    c,
    check_full,
    check_empty
)

from edpyt.matvec_product import (
    matvec_operator
)

from edpyt.build_mb_ham import (
    build_mb_ham
)

from edpyt.lanczos import (
    build_sl_tridiag
)

from edpyt.tridiag import (
    eigh_tridiagonal,
    gs_tridiag
)

from edpyt.espace import (
    build_empty_sector
)


def continued_fraction(a, b):
    sz = a.size
    def inner(e, eta, n=sz):
        if n==1:
            return b[sz-n] / (e + 1.j*eta - a[sz-n])
        else:
            return b[sz-n] / (e + 1.j*eta - a[sz-n] - inner(e, eta, n-1))
    inner.a = a
    inner.b = b
    return inner


def spectral(l, q):
    @vectorize('complex64(float64,float64)',nopython=True)
    def inner(e, eta):
        res=0.+0.j
        for i in prange(q.size):
            res += q[i] / ( e + 1.j*eta - l[i] )
        return res
    return inner


_reprs = ['cf','sp']


class Gf:
    def __init__(self, inner):
        self.funcs = []
        self.Z = np.inf
        self.inner = inner
    def add(self, a, b):
        self.funcs.append(
            self.inner(a, b)
        )
    def __call__(self, e, eta):
        out = np.sum([f(e, eta) for f in self.funcs], axis=0)
        return out / self.Z


def project(iI, pos, op, check_occupation, sctI, sctJ):
    """Project state iL of sector sctI onto eigenbasis of sector sctJ.

    """
    #
    # v[iM] = < (N+-1) | op(i) | N  >
    #       |        iM            iI
    #       |
    #       |             ___
    #       |             \
    #       = < (N+-1) |        c       op(i) | N  >
    #                iM   /__    iL,iI           iL
    #                        iL
    v0 = np.zeros(sctJ.d)
    idwI = np.arange(sctI.dwn) * sctI.dup
    idwJ = np.arange(sctJ.dwn) * sctJ.dup
    for iupI in range(sctI.dup):
        supI = sctI.states.up[iupI]
        # Check for empty impurity
        if check_occupation(supI, pos): continue
        sgnJ, supJ = op(supI, pos)
        iupJ = binsearch(sctJ.states.up, supJ)
        iL = iupI + idwI
        iM = iupJ + idwJ
        v0[iM] = np.float64(sgnJ)*sctI.eigvecs[iL,iI]
    return v0


def build_gf_coeff_sp(a, b, Ei=0., exponent=1., sign=1):
    """Returns Green function's coefficients for spectral.

    """
    #     ___               2
    #     \             q[r]                  -(beta E)
    # G =         -----------------------    e
    #     /__ r    z - sign * (En[r]-Ei)
    #
    En, Un = eigh_tridiagonal(a, b[1:])
    q = (Un[0] * b[0])**2 * exponent
    En -= Ei
    if sign<0:
        En *= -1.
    return En, q


def build_gf_coeff_cf(a, b, Ei=0., exponent=1., sign=1):
    """Returns Green function's coefficients for continued_fraction.

    """
    #     ___     -(beta E)         2
    #     \      e                b0
    # G =                -------------------        2
    #     /__ r           z - sign * (a0-Ei) -    b1
    #                                           ------
    #                                           z - sign * (a1-Ei)
    #
    b **= 2
    b[0] *= exponent
    a -= Ei
    if sign<0:
        a *= -1.
    return a, b


def build_gf_lanczos(H, V, espace, beta, egs=0., pos=0, mu=0., repr='cf'):
    """Build Green's function with exact diagonalization.

    """
    #
    #             ______
    #             \                                      _                                                      +               _
    #         1    \            (-beta (E(N)l - E0))    |      | < (N-1)l'| c(i)  | Nl > |        | < (N+1)l'| c(i)  | Nl > |    |
    # G  =   ---               e                        |    ------------------------------  +  ------------------------------   |
    # ii      Z    /                                    |_     iw   - ( E(N)l - E(N-1)l' )        iw   - ( E(N+1)l' - E(N)l )   _|
    #             /_____
    #            N,l,l'.
    #
    n = H.shape[0]
    irepr =_reprs.index(repr)
    gf_kernel = [continued_fraction, spectral][irepr]
    build_gf_coeff = [build_gf_coeff_cf, build_gf_coeff_sp][irepr]
    gf = Gf(gf_kernel)
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
        for iI in range(sctI.eigvals.size):
            exponent = np.exp(-beta*(sctI.eigvals[iI]-egs))
            # N+1 (one more up spin)
            nupJ = nupI+1
            ndwJ = ndwI
            # Cannot have more spin than spin states
            if nupJ <= n:
                # Arrival sector
                sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
                # < (N+1)l'| cdg(i)  | Nl >
                v0 = project(iI, pos, cdg, check_full, sctI, sctJ)
                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                gf.add(
                    *build_gf_coeff(aJ, bJ, sctI.eigvals[iI], exponent)
                )
            # N-1 (one more up spin)
            nupJ = nupI-1
            ndwJ = ndwI
            # Cannot have negative spin
            if nupJ >= 0:
                # Arrival sector
                sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
                # < (N-1)l'| c(i)  | Nl >
                v0 = project(iI, pos, c, check_empty, sctI, sctJ)
                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                aJ, bJ = build_sl_tridiag(matvec, v0)
                gf.add(
                    *build_gf_coeff(aJ, bJ, sctI.eigvals[iI], exponent, sign=-1)
                )

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-egs)).sum() for
        (nup, ndw), sct in espace.items())
    gf.Z = Z

    return gf
