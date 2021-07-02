import numpy as np

from edpyt.build_mb_ham import build_mb_ham
from edpyt.espace import build_empty_sector, build_from_sector, solve_sector
from edpyt.lanczos import build_sl_tridiag
from edpyt.lookup import binsearch
from edpyt.matvec_product import matvec_operator
from edpyt.operators import c, cdg, check_empty, check_full
from edpyt.operators import check_empty as not_empty
from edpyt.operators import check_full as not_full
from edpyt.tridiag import eigh_tridiagonal
from edpyt._continued_fraction import continued_fraction as _cfpyx
from edpyt.sector import OutOfHilbertError, get_cdg_sector, get_c_sector
from edpyt.gf_exact import project_exact_up, project_exact_dw


def continued_fraction(a, b):
    sz = a.size
    def inner(e, eta, n=sz):
        z = np.atleast_1d(e + 1.j*eta)
        return _cfpyx(z, a, b)
        # if n==1:
        #     return b[sz-n] / (e + 1.j*eta - a[sz-n])
        # else:
        #     return b[sz-n] / (e + 1.j*eta - a[sz-n] - inner(e, eta, n-1))
    inner.a = a
    inner.b = b
    return inner


def spectral(l, q):
    def inner(e, eta):
        z = np.atleast_1d(e + 1.j*eta)
        return np.dot(np.reciprocal(z[:,None]-l[None,:]),q)
        # np.einsum('i,ki',q,np.reciprocal(z[:,None]-l[None,:]))
    return inner


_reprs = ['cf','sp']


class Gf:
    def __init__(self):
        self.funcs = []
        self.Z = np.inf
    def add(self, kernel, a, b):
        self.funcs.append(
            kernel(a, b)
        )
    def __call__(self, e, eta):
        out = np.sum([f(e, eta) for f in self.funcs], axis=0)
        return out / self.Z


def project_up(pos, op, sctI, sctJ):
    """Project states of sector sctI onto eigenbasis of sector sctJ (nupJ=nupI+1).

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
    v0 = np.zeros((sctI.eigvals.size,sctJ.d))
    check_occupation = check_full if op is c else check_empty # if op is cdg
    for iupI in range(sctI.dup):
        supI = sctI.states.up[iupI]
        # Check for empty impurity
        if check_occupation(supI, pos):
            sgnJ, supJ = op(supI, pos)
            iupJ = binsearch(sctJ.states.up, supJ)
            v0[:,iupJ::sctJ.dup] = np.float64(sgnJ)*sctI.eigvecs[iupI::sctI.dup,:].T
    return v0


def project_dw(pos, op, sctI, sctJ):
    """Project states of sector sctI onto eigenbasis of sector sctJ (ndwJ=ndwI+1).

    """
    v0 = np.zeros((sctI.eigvals.size,sctJ.d))
    check_occupation = check_full if op is c else check_empty # if op is cdg
    for idwI in range(sctI.dwn):
        sdwI = sctI.states.dw[idwI]
        # Check for empty impurity
        if check_occupation(sdwI, pos):
            sgnJ, sdwJ = op(sdwI, pos)
            idwJ = binsearch(sctJ.states.dw, sdwJ)
            v0[:,idwJ*sctJ.dup:(idwJ+1)*sctJ.dup] = np.float64(sgnJ)*sctI.eigvecs[idwI*sctI.dup:(idwI+1)*sctI.dup,:].T
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


def build_gf_lanczos(H, V, espace, beta, egs=0., pos=0, repr='cf', ispin=0, separate=False):
    """Build Green's function with exact diagonalization.

    TODO : make it compatible with gf[spin] since spins may share
           same hilbert space but have two different onsites and hoppings (AFM).
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
    n = H.shape[-1]
    irepr =_reprs.index(repr)
    gf_kernel = [continued_fraction, spectral][irepr]
    build_gf_coeff = [build_gf_coeff_cf, build_gf_coeff_sp][irepr]
    project_exact = [project_exact_up, project_exact_dw][ispin]
    project = [project_up, project_dw][ispin]
    gfe = Gf()
    if separate:
        gfh = Gf()
    else:
        gfh = gfe
    #
    # Symbols Map:
    #    N -> I
    #  N+1 -> J
    #    l -> iI
    #  __
    # \
    #
    # /__ N,l
    # for l in range(sctI.eigvals.size)
    for nupI, ndwI in espace.keys():
        sctI = espace[(nupI,ndwI)]
        #  __
        # \
        #
        # /__ l
        exponents = np.exp(-beta*(sctI.eigvals-egs))
        try: # Add spin (N+1 sector)
            nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin)
        except OutOfHilbertError: # More spin than spin states
            pass
        else:
            # Arrival sector
            sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
            # solve with LAPACK
            if (sctJ.d <= 10):
                eigvals, eigvecs = solve_sector(
                    H, V, sctJ.states.up, sctJ.states.dw
                )
                sctJ = build_from_sector(sctJ,eigvals=eigvals,eigvecs=eigvecs)
                # <J|I>
                bJ = project_exact(pos, cdg, sctI, sctJ)
                # EJ-EI
                l = sctJ.eigvals[:,None] - sctI.eigvals[None,:]
                q = bJ**2 * exponents[None,:]
                gfe.add(
                    spectral,
                    l.reshape(-1,), q.reshape(-1,)
                )
            # solve with Lanczos
            else:
                # <I|J>
                v0 = project(pos, cdg, sctI, sctJ)
                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                for iL in range(sctI.eigvals.size):
                    aJ, bJ = build_sl_tridiag(matvec, v0[iL])
                    gfe.add(
                        gf_kernel,
                        *build_gf_coeff(aJ, bJ, sctI.eigvals[iL], exponents[iL])
                    )
        try: # Remove spin (N-1 sector)
            nupJ, ndwJ = get_c_sector(nupI, ndwI, ispin)
        except OutOfHilbertError: # Negative spin
            pass
        else:
            # Arrival sector
            sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
            # solve with LAPACK
            if (sctJ.d <= 10):
                eigvals, eigvecs = solve_sector(
                    H, V, sctJ.states.up, sctJ.states.dw
                )
                sctJ = build_from_sector(sctJ,eigvals=eigvals,eigvecs=eigvecs)
                # <J|I>
                bJ = project_exact(pos, c, sctI, sctJ)
                # EI-EJ
                l = - sctJ.eigvals[:,None] + sctI.eigvals[None,:]
                q = bJ**2 * exponents[None,:]
                gfh.add(
                    spectral,
                    l.reshape(-1,), q.reshape(-1,)
                )
            # solve with Lanczos
            else:
                # <I|J>
                v0 = project(pos, c, sctI, sctJ)
                matvec = matvec_operator(
                    *build_mb_ham(H, V, sctJ.states.up, sctJ.states.dw)
                )
                for iL in range(sctI.eigvals.size):
                    aJ, bJ = build_sl_tridiag(matvec, v0[iL])
                    gfh.add(
                        gf_kernel,
                        *build_gf_coeff(aJ, bJ, sctI.eigvals[iL], exponents[iL], sign=-1)
                    )

    # Partition function (Z)
    Z = sum(np.exp(-beta*(sct.eigvals-egs)).sum() for
        (nup, ndw), sct in espace.items())
    gfe.Z = Z
    gfh.Z = Z

    if separate:
        return gfe, gfh
    return gfe
