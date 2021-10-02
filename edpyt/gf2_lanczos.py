import numpy as np
from functools import lru_cache
from edpyt.gf_lanczos import *

# Not automatically imported by gf_lanczos
_reprs = ['cf','sp']

class _Gf2():
    def __init__(self, gf) -> None:
        self.n = gf.shape[0]
        self.gf = gf

    @staticmethod
    def evaluate(gf, e, eta):
        
        n = gf.shape[0]
        G = np.empty((n,n,e.size),complex)

        for ipos in range(n):
            G[ipos,ipos] = gf[ipos,ipos](e, eta)

        for ipos in range(n):
            for jpos in range(ipos+1,n):
                G[ipos,jpos] = 0.5*(gf[ipos,jpos](e,eta)-G[ipos,ipos]-G[jpos,jpos])
                G[jpos,ipos] = G[ipos,jpos]

        return G
    
    def __getitem__(self, ij):
        i,j = ij
        if i==j:
            return self.gf[i,j]
        return lambda e, eta: 0.5*(self.gf[i,j](e,eta)
                                   -self.gf[i,i](e,eta)
                                   -self.gf[j,j](e,eta))

    def __call__(self, e, eta):
        return self.evaluate(self.gf, e, eta)
    
class Gf2:
    
    def __new__(cls, gfe, gfh=None):
        if gfh is None:
            return _Gf2(gfe)
        return super().__new__(cls)
    
    def __init__(self, gfe, gfh) -> None:
        self.Gf2e = _Gf2(gfe)
        self.Gf2h = _Gf2(gfh)
        
    def __getitem__(self, ij):
        i,j = ij
        return lambda e, eta: self.Gf2e[i,j](e,eta) + self.Gf2h[i,j](e,eta)
        
    def __call__(self, e, eta):
        return self.Gf2e(e, eta), self.Gf2h(e, eta)


def build_gf_offdiag(H, V, espace, beta, ipos, jpos, ispin=0, egs=0., repr='cf', separate=False):
    """Build off-diagonal element of the + green's function."""
    # + green's function (see https://arxiv.org/pdf/0806.2690.pdf Eq. 33)
    
    n = H.shape[-1]
    irepr = _reprs.index(repr)
    gf_kernel = [continued_fraction, spectral][irepr]
    build_gf_coeff = [build_gf_coeff_cf, build_gf_coeff_sp][irepr]
    project = [project_up, project_dw][ispin]
    gfe = Gf()
    if separate:
        gfh = Gf()
    else:
        gfh = gfe

    for nupI, ndwI in espace.keys():
        sctI = espace[(nupI,ndwI)]
        exponents = np.exp(-beta*(sctI.eigvals-egs))

        try: # Add spin (N+1 sector)
            nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin)
        except OutOfHilbertError: # More spin than spin states
            pass
        else:
            sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
            #              +        +
            # Apply      c     + c
            #             i,s      i's'
            v0 = project(ipos, n, cdg, sctI, sctJ) + project(jpos, n, cdg, sctI, sctJ)
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
            #
            # Apply      c     + c
            #             i,s      i's'
            v0 = project(ipos, n, c, sctI, sctJ) + project(jpos, n, c, sctI, sctJ)
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


def build_gf2_lanczos(H, V, espace, beta, egs=0., ispin=0, repr='cf', separate=False):

    n = H.shape[-1]
    gfe = np.ndarray((n,n),object)
    gfh = None
    
    if separate:
        gfh = np.ndarray((n,n),object)
        def add(i, j, gf):
            gfe[i,j] = gf[0]
            gfh[i,j] = gf[1]
    else:
        def add(i, j, gf):
            gfe[i,j] = gf

    #   ss
    # G
    #   ii    
    for ipos in range(n):
        gf = build_gf_lanczos(
            H, V, espace, beta, egs, ipos, repr, ispin, separate)
        add(ipos,ipos,gf)
    #   ss
    # G
    #   ii' 
    for ipos in range(n):
        for jpos in range(ipos+1,n):
            gf = build_gf_offdiag(
                H, V, espace, beta, ipos, jpos, ispin, egs, repr, separate)
            add(ipos,jpos,gf)

    return Gf2(gfe, gfh)