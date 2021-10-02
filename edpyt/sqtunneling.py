import numpy as np

from edpyt.sector import (
    OutOfHilbertError, get_c_sector, get_cdg_sector)

from edpyt.gf_exact import (
    project_exact_up,
    project_exact_dw)

from edpyt.operators import cdg, c


def project_add(ispin, n, nupI, ndwI, espace):
    """Projec to particle sector with +1 `ispin` electron.
    """
    #    i,s               +
    #   y        =  < m+|c   | n >  nF(E-mu)
    #    mn               i,s
    nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin) # |m+>
    sctI = espace[nupI,ndwI]
    sctJ = espace[nupJ,ndwJ]
    project_exact = project_exact_up if ispin==0 else project_exact_dw
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    for i in range(n):
        v_JI[...,i] = project_exact(i, n, cdg, sctI, sctJ)
    E = sctJ.eigvals[:,None]-sctI.eigvals[None,:]
    return [Gfe((nupJ,ndwJ,j),(nupI,ndwI,i),v_JI[j,i],E[j,i]) for j,i in np.ndindex(v_JI.shape[:2])]


def project_sub(ispin, n, nupI, ndwI, espace):
    """Projec to particle sector with -1 `ispin` electron."""
    #    i,s               -
    #   y        =  < m-|c   | n >  ( 1 - nF(E-mu) )
    #    mn               i,s
    nupJ, ndwJ = get_c_sector(nupI, ndwI, ispin) # |m->
    sctI = espace[nupI,ndwI]
    sctJ = espace[nupJ,ndwJ]
    project_exact = project_exact_up if ispin==0 else project_exact_dw
    v_JI = np.empty((sctJ.eigvals.size,sctI.eigvals.size,n))
    for i in range(n):
        v_JI[...,i] = project_exact(i, n, c, sctI, sctJ)
    E = -sctJ.eigvals[:,None]+sctI.eigvals[None,:]
    return [Gfh((nupJ,ndwJ,j),(nupI,ndwI,i),v_JI[j,i],E[j,i]) for j,i in np.ndindex(v_JI.shape[:2])]


getdns = lambda dnu,dnd: (dnu+dnd, (dnu-dnd)//(dnu+dnd))
getdns.__doc__ = """
    Get change in total # of electrons and spin given 
    the change in up and down components. This function
    is the inverse of `getdud`."""

getdud = lambda dn,spin: (dn*(spin+1)//2, dn*(-spin+1)//2)
getdud.__doc__ =  """
    Get change in in up and down components given the change in 
    total # of electrons and spin. This function
    is the inverse of `getdns`."""


def project_sector(n, nupI, ndwI, espace, gfdict, dns=None):
    if dns is None:
        dns = [getdud(dn,spin) for dn in [-1,1] for spin in [1,-1]]
    for dnu,dnd in dns:
        dn, spin = getdns(dnu,dnd)
        ispin = (-spin+1)//2 # map spin:ispin={1:0,-1,1}
        if dn>0:
            try:
                gflist = project_add(ispin, n, nupI, ndwI, espace)
            except OutOfHilbertError:
                continue
        else:
            try:
                gflist = project_sub(ispin, n, nupI, ndwI, espace)
            except OutOfHilbertError:
                continue
        for gf in gflist:
            gfdict[(gf.idF,gf.idI)] = gf


def build_transition_elements(n, egs, espace, cutoff=None):
    """Build projector space. Compute all possible matrix elements connecting 
    the ground state and the vectors reached by the latter (including the matrix
    elements connecting the latter vectors).
    """
    ngs = [ns for ns,sct in espace.items() if abs(sct.eigvals.min()-egs)<1e-9]
    dns = [getdud(dn,spin) for dn in [-1,1] for spin in [1,-1]]
    reached_by_gs = [(nup+dnu,ndw+dnd) for nup,ndw in ngs for dnu,dnd in dns]
    active_sectors = list(filter(lambda ns: all((i>=0)&(i<=n) for i in ns), reached_by_gs)) + ngs
    gfdict = {}
    for nup,ndw in np.unique(active_sectors,axis=0):
        dns_subset = [(dnu,dnd) for dnu,dnd in dns if (nup+dnu,ndw+dnd) in active_sectors]
        project_sector(n, nup, ndw, espace, gfdict, dns_subset)
    if cutoff is not None:
        return screen_transition_elements(gfdict, egs, espace, cutoff)
    return gfdict


class _Gf:
    """Green's function.
    
    Args:
        E : (E+ - En') or (En' - E-)
        v : <m-|c|n'> or <m+|c+|n'>
    """
      
    def __init__(self, idJ, idI, v, E) -> None:
        self.idF = idJ
        self.idI = idI
        self.v = v
        self.E = E
    
    @staticmethod
    def nF(E, beta):
        """Fermi distribution."""
        return 1/(np.exp(beta*E)+1)
    
    def __call__(self, a):
        return (self.v.dot(a))**2


class Gfe(_Gf):
    """Electron.
        |<m+|c+|n'>|^2 nF(E-mu)
    """
    def __call__(self, a, beta, mu):
        return super().__call__(a)*self.nF(self.E-mu,beta)


class Gfh(_Gf):
    """Hole.
        |<m-|c+|n'>|^2 (1-nF(E-mu))
    NOTE: nF(-E)==(1-nF(E))
    """
    def __call__(self, a, beta, mu):
        return super().__call__(a)*self.nF(-self.E+mu,beta)


def build_rate_matrix(gfdict, beta, mu, A):
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
    sz, odd = np.divmod(len(gfdict), 2)
    assert ~odd, """
        Invalid gf list. Each matrix element must contain 
        its complex conjugate.
    """
    idF = sorted(set([idF for idF,_ in gfdict.keys()]))
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    W = np.zeros((sz, sz))
    for lead in range(2):
        for i, f in np.ndindex(sz, sz):
            if i != f:
                try:
                    gf = gfdict[map[f],map[i]]
                    gamma = gf(A[lead], beta, mu[lead])
                except:
                    continue
                W[i,i] -= gamma
                W[f,i] += gamma
    return W


def build_transition_matrix(gfdict, beta, mu, a):
    #                __                       __  T   _   _
    #  |     |      |                           |    |     |     
    #  |     |      |              _            |    |     | 
    #  | |0> |      |    0      - |        --   |    |  P  |     
    #  |     |      |               01          |    |   0 | 
    #  |     |      |                           |    |     |
    #  |     |  =   |     _                 --  |    |     |      
    #  | |1> |      |  + |          0           |    |  P  |      
    #  |     |      |      10                   |    |   1 | 
    #  |     |      |                           |    |     |     
    #  |  :  |      |       :       :      \    |    |  :  |  
    sz, odd = np.divmod(len(gfdict), 2)
    assert ~odd, """
        Invalid gf list. Each matrix element must contain 
        its complex conjugate.
    """
    idF = sorted(set([idF for idF, _ in gfdict.keys()]))
    map = {i:id for i,id in enumerate(idF)}
    sz = len(idF)
    T = np.zeros((sz, sz))
    for i, f in np.ndindex(sz, sz):
        if i != f:
            try:
                gf = gfdict[map[f],map[i]]
                if isinstance(gf, Gfe):
                    gamma = gf(a, beta, mu)
                elif isinstance(gf, Gfh):
                    gamma = - gf(a, beta, mu)
                else:
                    raise RuntimeError(f"Green function (type(gf)) not recognized.")
            except:
                continue
            T[f,i] = gamma
    return T

    
def screen_transition_elements(gfdict, egs, espace, cutoff):
    return {(idF, idI):gf for (idF, idI),gf in gfdict.items() 
            if (abs(gf.E)<cutoff)
            and(abs(espace[idF[:2]].eigvals[idF[2]]-egs)<cutoff)
            and(abs(espace[idI[:2]].eigvals[idI[2]]-egs)<cutoff)}
    
    
def find_active_sectors(n, ngs):
    # ngs = [ns for ns,sct in espace.items() if abs(sct.eigvals.min()-egs)<1e-9]
    dns = [getdud(dn,spin) for dn in [-1,1] for spin in [1,-1]]
    reached_by_gs = [(nup+dnu,ndw+dnd) for nup,ndw in ngs for dnu,dnd in dns]
    active_sectors = list(filter(lambda ns: all((i>=0)&(i<=n) for i in ns), reached_by_gs)) + ngs
    return active_sectors


def get_active_neig(n, ngs, val=2):
    from edpyt.lookup import get_sector_index
    neig = np.zeros((n+1)*(n+1),int)
    idx = [get_sector_index(n,nup,ndw) for nup,ndw in find_active_sectors(n,ngs)]
    for i in idx:
        neig[i] = val
    return neig