import numpy as np
from functools import lru_cache
from gf_lanczos import *


class sGf():
    def __init__(self, n, gfe, gfh) -> None:
        self.n = n
        self.gfe = gfe
        self.gfe = gfh
        # self.gfh.__call__ = lru_cache(self.gfh.__call__)
        # self.gfe.__call__ = lru_cache(self.gfe.__call__)

    def build(self, e, eta):
        n = self.n
        gfe = np.empty((2,2,n,n),complex)
        gfh = np.empty((2,2,n,n),complex)

        for ispin in range(2):
            for ipos in range(n):
                gfe[ispin,ispin,ipos,ipos] = self.gfe[ispin,ispin,ipos,ipos](e, eta)
                gfh[ispin,ispin,ipos,ipos] = self.gfh[ispin,ispin,ipos,ipos](e, eta)

        for ispin, jspin in np.ndindex(2,2):
            if ispin==jspin: continue
            for ipos in range(n):
                gfe[ispin,jspin,ipos,ipos] = 0.5*(
                    self.gfe[ispin,jspin,ipos,ipos](e, eta)
                    - gfe[ispin,ispin,ipos,ipos]
                    - gfe[jspin,jspin,ipos,ipos]
                )
                gfh[ispin,jspin,ipos,ipos] = 0.5*(
                    self.gfh[ispin,jspin,ipos,ipos](e, eta)
                    - gfh[ispin,ispin,ipos,ipos]
                    - gfh[jspin,jspin,ipos,ipos]
                )

        for ispin in range(2):
            for ipos, jpos in np.ndindex(n,n):
                if ipos==jpos: continue
                gfe[ispin,ispin,ipos,jpos] = 0.5*(
                    self.gfe[ispin,ispin,ipos,jpos](e, eta)
                    - gfe[ispin,ispin,ipos,ipos]
                    - gfe[ispin,ispin,jpos,jpos]
                )
                gfh[ispin,ispin,ipos,jpos] = 0.5*(
                    self.gfh[ispin,ispin,ipos,jpos](e, eta)
                    - gfh[ispin,ispin,ipos,ipos]
                    - gfh[ispin,ispin,jpos,jpos]
                )

        for ispin, jspin in np.ndindex(2,2):
            if ispin==jspin: continue
            for ipos, jpos in np.ndindex(n,n):
                if ipos==jpos: continue
                gfe[ispin,jspin,ipos,jpos] = 0.5*(
                    self.gfe[ispin,jspin,ipos,jpos](e, eta)
                    - gfe[ispin,ispin,ipos,ipos]
                    - gfe[jspin,jspin,jpos,jpos]
                )
                gfe[ispin,jspin,ipos,jpos] = 0.5*(
                    self.gfh[ispin,jspin,ipos,jpos](e, eta)
                    - gfh[ispin,ispin,ipos,ipos]
                    - gfh[jspin,jspin,jpos,jpos]
                )

        self.gfe_ = gfe
        self.gfh_ = gfh

    def __call__(self, e, eta):
        self.build(e, eta)
        return self.gfe_ + self.gfh_

    @property
    def e(self, ispin, jspin, ipos, jpos):
        return self.gfe_[ispin,jspin,ipos,jpos]
    
    @property
    def h(self, ispin, jspin, ipos, jpos):
        return self.gfh_[ispin,jspin,ipos,jpos]
    


def build_gf_offdiag(H, V, espace, beta, ipos, jpos, ispin, jspin, egs=0., repr='cf'):
    n = H.shape[-1]
    irepr =_reprs.index(repr)
    gf_kernel = [continued_fraction, spectral][irepr]
    build_gf_coeff = [build_gf_coeff_cf, build_gf_coeff_sp][irepr]
    iproj = [project_up, project_dw][ispin]
    jproj = [project_up, project_dw][jspin]
    gfe = Gf()
    gfh = Gf()

    for nupI, ndwI in espace.keys():
        sctI = espace[(nupI,ndwI)]
        exponents = np.exp(-beta*(sctI.eigvals-egs))

        try: # Add spin (N+1 sector)
            nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin)
        except ValueError: # More spin than spin states
            pass
        else:
            sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
            #              +        +
            # Apply      c     + c
            #             i,s      i's'
            v0 = iproj(ipos, cdg, not_full, sctI, sctJ) + jproj(jpos, cdg, not_full, sctI, sctJ)
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
        except ValueError: # Negative spin
            pass
        else:
            # Arrival sector
            sctJ = espace.get((nupJ, ndwJ), None) or build_empty_sector(n, nupJ, ndwJ)
            #
            # Apply      c     + c
            #             i,s      i's'
            v0 = iproj(ipos, c, not_empty, sctI, sctJ) + jproj(jpos, c, not_empty, sctI, sctJ)
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

    return gfe, gfh


def build_gf2_lanczos(H, V, espace, beta, egs=0., pos=0.):

    gfe = [None]*4
    gfh = [None]*4
    n = H.shape[-1]

    #   ss
    # G
    #   ii    
    for ispin in range(2):
        for ipos in range(n):
            gfe[ispin,ispin,ipos,ipos], gfh[ispin,ispin,ipos,ipos] = build_gf_lanczos(
                H, V, espace, beta, egs, ipos, ispin, separate=True)

    #   ss'
    # G
    #   ii' 
    for ispin, jspin in np.ndindex(2,2):
        for ipos, jpos in np.ndindex(n,n):
            gfe[ispin,jspin,ipos,jpos], gfh[ispin,jspin,ipos,jpos] = build_gf_offdiag(
                H, V, espace, ipos, ispin, jpos, jspin)


    return sGf(n, gfe, gfh)