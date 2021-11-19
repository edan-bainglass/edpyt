from warnings import warn
import numpy as np
from edpyt import eigh_arpack as sla
from collections import namedtuple
from dataclasses import make_dataclass, field
from itertools import product

from edpyt.sector import (
    generate_states,
    get_sector_dim,
    binom
)

from edpyt.build_mb_ham import (
    build_mb_ham
)


from edpyt.matvec_product import (
    todense,
    matvec_operator
)

from edpyt.lookup import (
    get_sector_index
)


SzStates = namedtuple('States',['up','dw'])
Sector = make_dataclass('Sector', ['states', ('d', int),
                         ('eigvals', np.ndarray, field(default=None)),
                         ('eigvecs', np.ndarray, field(default=None))])


def build_empty_sector(n, *p):
    """Build sector.
    
    Args:
        n : # of sites
        p : # of particles.
            - integer : total # of electrons.
            - (nup, ndw) : up & down # of electrons.
    """
    if len(p) == 1:
        states = generate_states(2*n, p[0])
        d = states.size # Hilbert dimension.
    else:
        nup, ndw = p
        states = SzStates(
            generate_states(n, nup),        
            generate_states(n, ndw))
        d = states.up.size * states.dw.size # Hilbert dimension.        
    return Sector(states, d)


def solve_sector(H, V, sct, k=None):
    """Diagonalize sector.

    """
    if k is None: k = sct.d
    if (k == sct.d) or (sct.d <= 512):
        eigvals, eigvecs = _solve_lapack(H, V, sct)
        if k<sct.d: eigvals, eigvecs = eigvals[:k], eigvecs[:,:k]
    else:
        eigvals, eigvecs = _solve_arpack(H, V, sct, k)
    return eigvals, eigvecs


def _solve_lapack(H, V, sct):
    """Diagonalize sector with LAPACK.

    """
    ham = todense(
        *build_mb_ham(H, V, sct)
    )
    return np.linalg.eigh(ham)


def _solve_arpack(H, V, sct, k=6):
    """Diagonalize sector with ARPACK.

    """
    matvec = matvec_operator(
        *build_mb_ham(H, V, sct)
    )
    return sla.eigsh(sct.d, k, matvec)


def get_espace_dim(n, neig_max=None, symmetry='sz'):
    """Get dimensions of sectors."""
    if symmetry.lower() == 'sz':
        neig_sector = np.zeros((n+1)*(n+1), int)
        for nup, ndw in np.ndindex(n+1,n+1):
            neig_sector[get_sector_index(n,nup,ndw)] = get_sector_dim(n,nup,ndw)
    
    elif symmetry.upper() == 'N':
        neig_sector = np.zeros(2*n+1, int)
        for ndu in np.ndindex(2*n+1):
            neig_sector[ndu] = binom(2*n,ndu)
    
    else:
        raise NotImplementedError(f"Symmetry - {symmetry} - non implemented.")
    
    if neig_max:
        # clip to max. # of eigenvalues.
        neig_sector = np.where(neig_sector <= neig_max, neig_sector, neig_max)
        
    return neig_sector


def _sz_iter_sectors(n):
    """Iterate over all (sz-)sectors.
    Return:
        (quantum numbers, states, size).
    """
    for nup in range(n+1):
        states_up = generate_states(n, nup)
        for ndw in range(n+1):
            states_dw = generate_states(n, ndw)
            states = SzStates(states_up, states_dw)
            d = states_up.size*states_dw.size
            yield (nup,ndw), Sector(states, d)


def _N_iter_sectors(n):
    """Iterate over all (N-)sectors.
    Return:
        (quantum number, states, size).
    """
    for ndu in range(2*n+1):
        states = generate_states(2*n, ndu)
        yield (ndu,), Sector(states, states.size)


def build_espace(H, V, neig_sector=None, symmetry='sz'):
    """Generate and solve all sectors in hilbert space."""
    n = H.shape[-1]
    
    if symmetry.lower() == 'sz':
        iter_sectors = _sz_iter_sectors
        get_sector_index = lambda qns: qns[0]*(n+1) + qns[1]
    
    elif symmetry.upper() == 'N':
        iter_sectors = _N_iter_sectors
        get_sector_index = lambda qns: qns[0]
    
    else:
        raise NotImplementedError(f"Symmetry - {symmetry} - non implemented.")
    
    espace = dict()
    if neig_sector is None:
        neig_sector = get_espace_dim(n, symmetry=symmetry)

    egs = np.inf
        
    for qns, sct in iter_sectors(n):
        neig = neig_sector[get_sector_index(qns)]
        if neig == 0:
            continue
        # Diagonalize!
        sct.eigvals, sct.eigvecs = solve_sector(H, V, sct, neig)
        if sct.eigvals.size==0:
            warn(f'Zero-size eigenvalues for sector with quantum numbers {qns}.')
            continue
        
        espace[qns] = sct

        # Update GS energy
        egs = min(sct.eigvals.min(), egs)

    return espace, egs


def build_non_interacting_espace(ek):
    """Build spetrum of non-interacting paricles.
    
    Args:
        ek : (np.ndarray, ndim=1) 
            on-site particle energies.
    Returns:
        espace : non-interacting spectrum
    """
    from edpyt.build_mb_ham import add_onsites
    espace = dict()
    egs = np.inf

    n = ek.size
    Uk = np.zeros_like(ek)
    ek = np.tile(ek, (2,n))
    
    for qns, sct in _sz_iter_sectors(n):
        states = sct.states
        eigvals = np.empty(sct.d)
        add_onsites(ek, Uk, states.up, states.dw, eigvals, 0.)
        eigvals.sort()
        sct.eigvals = eigvals
        espace[qns] = sct
        egs = min(eigvals.min(), egs)
    return espace, egs


def screen_espace(espace, egs, beta=1e6, cutoff=1e-9):#, neig_sector=None, n=0):
    """Keep sectors containing relevant eigen-states:
    any{ exp( -beta * (E(N)-egs) ) } > cutoff. If beta
    is > ~1e3, then only the GS is kept.

    """

    delete = []
    for qns, sct in espace.items():
        diff = np.exp(-beta*(sct.eigvals-egs)) > cutoff
        if diff.any():
            if (sct.eigvecs.ndim<2): 
                raise RuntimeError('sct.eigvecs.ndim < 2!')
            keep_idx = np.where(diff)[0]
            sct.eigvals = sct.eigvals[keep_idx]
            sct.eigvecs = sct.eigvecs[:,keep_idx]
            assert sct.eigvecs.ndim>1
        else:
            delete.append(qns)

    for qns in delete:
        espace.pop(qns)


def adjust_neigsector(espace, neig, n):
    """Adjust # of eigen-states to compute for each sector.
    
    WARNING: espace is assumed to have sorted keys.

    RULES:
        Increase if:
            (i) # of eigen-states per sector is already
                equal to neig to compute.

        Decrease if:
            (i) # of eigen-states per sector is less than
                neig to compute.
            (ii) sector is absent from espace.

    """
    def _increase(isct):
        neig[isct] = min(
            get_sector_dim(n,nup,ndw),
            neig[isct]+1
        )
    def _decrease(isct):
        neig[isct] = max(
            1,
            neig[isct]-1
        )     

    iter_keys = product(range(n+1),range(n+1))
    for nup, ndw in espace.keys(): # ASSUME SORTED!
        nup_, ndw_ = next(iter_keys)
        # While sector is absent
        while (nup != nup_) or (ndw != ndw_):
            isct = get_sector_index(n,nup_,ndw_)
            _decrease(isct)
            nup_, ndw_ = next(iter_keys)
        isct = get_sector_index(n,nup_,ndw_)
        eigvals_size = espace[(nup,ndw)].eigvals.size
        if eigvals_size == neig[isct]:
            _increase(isct)
        elif eigvals_size <  neig[isct]:
            _decrease(isct)
    # All (nup_,ndw_) pairs larger than max {(nup,ndw)}
    for nup_, ndw_ in iter_keys:    
        isct = get_sector_index(n,nup_,ndw_)
        _decrease(isct)