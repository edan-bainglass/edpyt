from numba import vectorize
import numpy as np
from itertools import chain

# from edpyt.lookup import binsearch

from edpyt.sector import (
    get_c_sector, get_cdg_sector)

from edpyt.gf_exact import (
    # <N+1|c+(i,up)|N>
    # <N|c-(i,up)|N+1>
    project_exact_up, 
    # <N+1|c+(i,dw)|N>
    # <N|c-(i,dw)|N+1>
    project_exact_dw)

from edpyt.operators import (
    check_empty as not_empty, 
    check_full as not_full,
    cdg, c)

from edpyt.integrate_gf import integrate_gf


@vectorize('float64(float64, float64)')
def abs2(r, i):
    return r**2 + i**2


def projector(ispin):
    if ispin==0:
        return project_exact_up
    return project_exact_dw


def project_add(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace):
    """Cotunneling rate for a lead electron to go from lead l' to lead l and
    the central region to go from state n' to state n, i.e.:
     
            l' -> l
            n' -> n     
    """
    #            ___                           
    #   +        \                                            +
    #  y (ss') =        A     A       < n| c    | m+ > < m+| c      | n' >
    #   nn'      /__     l,i    l',i'        i,s               i',s'
    #            ii'                            
    n = A.shape[1]
    nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin_j) # |m+>
    nupF, ndwF = get_c_sector(nupJ, ndwJ, ispin_i) # |n>
    sctI = espace[(nupI, ndwI)] # |n'>
    sctJ = espace[(nupJ, ndwJ)] # |m+>
    sctF = espace[(nupF, ndwF)] # |n>
    proj_ispin_j = projector(ispin_j) 
    proj_ispin_i = projector(ispin_i) 
    v_JI = np.zeros((sctJ.eigvals.size,sctI.eigvals.size))
    v_FJ = np.zeros((sctF.eigvals.size,sctJ.eigvals.size))
    # Because one assumes that only the charge is a conserved quantity
    # (and not Sz), the states are product of up and down comonponents: 
    # 
    #    |s> = |s>   ,  |s> 
    #             dw      up
    #
    # e.g.
    #
    #   |0,0,0,1,...1,0>, |0,1,0,1,...0,,0>
    #
    # Hence, one should include the fermionic sign coming from the number
    # of down spins when the operator acts on the up component.
    sgnI = (-1.)**ndwI if ispin_j==0 else 1. 
    sgnJ = (-1.)**ndwJ if ispin_i==0 else 1.
    A2 = A**2
    pos_cdg = np.where(A[lead_extract]>1e-18)[0]
    pos_c = np.where(A[lead_inject]>1e-18)[0]
    for pos_j in pos_cdg:
        v_JI += A2[lead_extract,pos_j] * sgnI * proj_ispin_j(pos_j, cdg, not_full, sctI, sctJ)
        for pos_i in pos_c:
            v_FJ += A2[lead_inject,pos_i] * sgnJ * proj_ispin_i(pos_i, c, not_empty, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = sctJ.eigvals[:,None]-sctI.eigvals[None,:]
    return Gf2(E, v_FJ, v_JI), dE


def project_sub(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace):
    """Cotunneling rate for a lead electron to go from lead l to lead l' and
    the central region to go from state n' to state n, i.e.:
    
            l' -> l
            n' -> n    
    """
    #            ___                           
    #   -        \                           +                 
    #  y (ss') =        A     A       < n| c    | m- > < m-| c      | n' >
    #   nn'      /__     l',i    l,i'        i,s               i',s'
    #            ii'                           
    n = A.shape[1]
    nupJ, ndwJ = get_c_sector(nupI, ndwI, ispin_j) # |m+>
    nupF, ndwF = get_cdg_sector(n, nupJ, ndwJ, ispin_i) # |n>
    sctI = espace[(nupI, ndwI)] # |n'>
    sctJ = espace[(nupJ, ndwJ)] # |m+>
    sctF = espace[(nupF, ndwF)] # |n>
    proj_ispin_j = projector(ispin_j) 
    proj_ispin_i = projector(ispin_i) 
    v_JI = np.zeros((sctJ.eigvals.size,sctI.eigvals.size))
    v_FJ = np.zeros((sctF.eigvals.size,sctJ.eigvals.size))
    # See notes for add projection.
    sgnI = (-1.)**ndwI if ispin_j==0 else 1.
    sgnJ = (-1.)**ndwJ if ispin_i==0 else 1.
    A2 = A**2
    pos_c = np.where(A[lead_inject]>1e-18)[0]
    pos_cdg = np.where(A[lead_extract]>1e-18)[0]
    for pos_j in pos_c:
        v_JI += A2[lead_inject,pos_j] * sgnI * proj_ispin_j(pos_j, c, not_empty, sctI, sctJ)
        for pos_i in pos_cdg:
            v_FJ += A2[lead_extract,pos_i] * sgnJ * proj_ispin_i(pos_i, cdg, not_full, sctJ, sctF)
    dE = sctF.eigvals[:,None]-sctI.eigvals[None,:]
    E = -sctJ.eigvals[:,None]+sctI.eigvals[None,:]
    return Gf2(E, v_FJ, v_JI), dE


def project(A, lead_extract, lead_inject, nupI, ndwI, espace):
    """Cotunneling rate from state n to state n'.

    NOTE: The leads' and spin's indices are interchanged. For leads' indices
    this is because we want the sum l' -> l + l' <- l. For spin, one can show
    that to obtain the same final state n, the spin creation and annihilation
    orders have to be swaped.
    
    Example:

    |n'> = u,d
    |n> = u+1,d-1
    
        |    u+     d-   |    d-      u+
    ---------------------|-----------------
    u,d | u+1,d  u+1,d-1 |  u,d-1  (u+1,d-1)   , same final state   
    """
    n = A.shape[1]
    ispin = [(0,0),(0,1),(1,0),(1,1)]
    _Sigma = dict.fromkeys(ispin)
    dE = dict.fromkeys(ispin)
    for ispin_i, ispin_j in ispin:
        # Electron and hole green functions.
        try:
            Gf2e, dEe = project_add(A, lead_extract, lead_inject, ispin_i, ispin_j, nupI, ndwI, espace)
        except ValueError: # Out of Hilbert
            Gf2e, dEe = None, None
        try:
            Gf2h, dEh = project_sub(A, lead_extract, lead_inject, ispin_j, ispin_i, nupI, ndwI, espace)
        except ValueError: # Out of Hilbert
            Gf2h, dEh = None, None
        if (dEe is not None) and (dEh is not None):
            assert np.allclose(dEe, dEh), """
                Invalid projections. Initial and final states 
                after N+1 and N-1 projections are not the same."""
        dE[ispin_i, ispin_j] = dEh if dEe is None else dEe
        _Sigma[ispin_i, ispin_j] = Sigma(Gf2e, Gf2h)
    if (dE[(0,0)] is not None) and (dE[(1,1)] is not None):
        assert np.allclose(dE[(0,0)], dE[(1,1)]), """
            Invalid projections. Initial and final states 
            for (0,0) and (1,1) spin projections are not the same."""
    return _Sigma, dE


class Gf2:

    def __init__(self, E, v_FJ, v_JI) -> None:
        self.E = E
        self.v_FJ = v_FJ
        self.v_JI = v_JI

    @property
    def shape(self):
        return (self.v_FJ.shape[0],self.E.shape[0],self.v_JI.shape[1])

    def __call__(self, z):
        z = np.atleast_1d(z)
        if z.ndim == 1:
            z = z[:,None,None]
            res = np.einsum('ij,kjl,jl->kil',
                self.v_FJ,
                np.reciprocal(z-self.E[None,...]),
                self.v_JI
            )
        elif z.ndim == 2:
            z = z[None,...]
            res = np.einsum('ij,jil,jl->il',
                self.v_FJ,
                np.reciprocal(z-self.E[:,None,:]),
                self.v_JI
            )
        return res


class Sigma:

    def __init__(self, Gf2e, Gf2h) -> None:
        self.Gf2e = Gf2e
        self.Gf2h = Gf2h

    def __call__(self, z):
        res = self.Gf2e(z) + self.Gf2h(z)
        return abs2(res.real, res.imag)


# https://www.weizmann.ac.il/condmat/oreg/sites/condmat.oreg/files/uploads/Thesises/carmiphd.pdf


from mpmath import psi
from scipy.constants import hbar as _hbar, eV
hbar = _hbar/eV # Units of electron volt.

# Helpers

def _Psi(n, a, b, beta):
    """Polygamma function of order n."""
    return psi(n, 0.5 + (1.j*beta)/(2.*np.pi)*(a-b))

def _nB(eps, beta):
    """Bose distribution."""
    return 1./(np.exp(beta*eps)-1)

from warnings import warn

def _I1(C, eps, mu, beta):
    f = C**2 * beta/(2*np.pi) * np.imag(
        _Psi(1,mu[1],eps,beta) 
      - _Psi(1,mu[0],eps,beta)
    )
    with np.errstate(divide='raise',over='ignore'): # 1/0.
        try:
            w = _nB(mu[1]-mu[0],beta)
        except FloatingPointError as e:
            if abs(f)<1e-18: # 0./0. -> 1.
                return 1.
            else:
                raise e
    return w * f

def _I2(A, B, epsA, epsB, mu, beta):
    f = A*B * np.real(
        _Psi(0,epsA,mu[1],beta) 
      - _Psi(0,epsA,mu[0],beta) 
      - _Psi(0,epsB,mu[1],beta)
      + _Psi(0,epsB,mu[0],beta)
    )
    with np.errstate(divide='raise',over='ignore'): # 1./0.
        try:
            w = _nB(mu[1]-mu[0],beta)
            dAB_inv = 1. / (epsA - epsB)
        except FloatingPointError as e:
            if abs(f)<1e-18: # 0./0.
                return 1.
            else:
                raise e
    return w * dAB_inv * f

#

Gamma1 = _I1


def Gamma2(A, B, epsA, epsB, mu, beta):
    return _I1(A,epsA,mu,beta) \
         + _I1(B,epsB,mu,beta) \
         + _I2(A,B,epsA,epsB,mu,beta)


def J(Sigma, dE, P, beta, mu):
    """Density current."""
    res = 0.
    for sigma, de in zip(Sigma.values(), dE.values()):
        gf2e = sigma.Gf2e
        gf2h = sigma.Gf2h
        if (gf2e is not None) and (gf2h is not None):
            for f, i in np.ndindex(de.shape):
                res += Gamma2(gf2e.v_FJ[f,0]*gf2e.v_JI[0,i], #A
                       gf2h.v_FJ[f,0]*gf2h.v_JI[0,i], #B
                       gf2e.E[0,i], #epsA
                       gf2h.E[0,i], #epsB
                       [mu[0]-de[f,i],mu[1]], 
                       beta) * P[i]
        elif gf2e is not None:
            for f, i in np.ndindex(de.shape):
                res += Gamma1(gf2e.v_FJ[f,0]*gf2e.v_JI[0,i], #A
                       gf2e.E[0,i], #epsA
                       [mu[0]-de[f,i],mu[1]], 
                       beta) * P[i]
        else:
            for f, i in np.ndindex(de.shape):
                res += Gamma1(gf2h.v_FJ[f,0]*gf2h.v_JI[0,i], #B
                       gf2h.E[0,i], #epsB
                       [mu[0]-de[f,i],mu[1]], 
                       beta) * P[i]
    return res
