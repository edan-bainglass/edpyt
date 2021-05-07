import numpy as np
from numba import njit
from scipy import linalg as la
# from scipy.sparse import linalg as sla
from edpyt import eigh_arpack as sla
from collections import namedtuple, defaultdict
from dataclasses import make_dataclass
from dataclasses import replace as build_from_sector
from itertools import product

from edpyt.sector import (
    generate_states,
    get_sector_dim
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


States = namedtuple('States',['up','dw'])
# Sector = namedtuple('Sector',['states','d','dup','dwn','eigvals','eigvecs'])
Sector = make_dataclass('Sector',['states','d','dup','dwn','eigvals','eigvecs'])


def build_empty_sector(n, nup, ndw):
    states_up = generate_states(n, nup)
    states_dw = generate_states(n, ndw)
    return Sector(
        States(states_up, states_dw),
        states_up.size*states_dw.size,
        states_up.size,
        states_dw.size,
        None,
        None
    )


def solve_sector(H, V, states_up, states_dw, k=None):
    """Diagonalize sector.

    """
    d = states_up.size*states_dw.size
    if k is None: k = d
    if (k == d) or (d <= 10):
        eigvals, eigvecs = _solve_lapack(H, V, states_up, states_dw, k)
        if k<d: eigvals, eigvecs = eigvals[:k], eigvecs[:,:k]
    else:
        eigvals, eigvecs = _solve_arpack(H, V, states_up, states_dw, k)
    return eigvals, eigvecs


def _solve_lapack(H, V, states_up, states_dw, k):
    """Diagonalize sector with LAPACK.

    """
    ham = todense(
        *build_mb_ham(H, V, states_up, states_dw)
    )
    return la.eigh(ham, overwrite_a=True)


def _solve_arpack(H, V, states_up, states_dw, k=6):
    """Diagonalize sector with ARPACK.

    """
    matvec = matvec_operator(
        *build_mb_ham(H, V, states_up, states_dw)
    )
    return sla.eigsh(states_up.size*states_dw.size, k, matvec)
    # return sla.eigsh(matvec, k, which='SA')


def get_espace_dim(n):
    neig_sector = np.zeros((n+1)*(n+1),int)
    for nup, ndw in np.ndindex(n+1,n+1):
        neig_sector[get_sector_index(n,nup,ndw)] = get_sector_dim(n,nup,ndw)
    return neig_sector


def build_espace(H, V, neig_sector=None):
    """Generate full spectrum.

    Args:
        neig_sector : # of eigen states in each sector.

    Return:
        eig_space : list of eigen states ordered by energy.

    Return:
        eig_space
    """
    n = H.shape[-1]

    espace = defaultdict(Sector)
    if neig_sector is None:
        neig_sector = get_espace_dim(n)

    # Fill in eigen states in eigen space
    egs = np.inf
    for nup in range(n+1):
        states_up = generate_states(n, nup)
        for ndw in range(n+1):
            # Sequential index sector.
            isct = get_sector_index(n,nup,ndw)
            if neig_sector[isct] == 0:
                continue
            states_dw = generate_states(n, ndw)
            # Diagonalize sector
            eigvals, eigvecs = solve_sector(H, V, states_up, states_dw, neig_sector[isct])
            espace[(nup,ndw)] = Sector(
                States(states_up, states_dw),
                states_up.size*states_dw.size,
                states_up.size,
                states_dw.size,
                eigvals,
                eigvecs
            )

            # Update GS energy
            egs = min(eigvals.min(), egs)

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
    espace = defaultdict(Sector)
    egs = np.inf

    n = ek.size
    Uk = np.zeros_like(ek)
    ek = np.tile(ek, (2,n))
    for nup in range(n+1):
        states_up = generate_states(n, nup)
        dup = states_up.size
        for ndw in range(n+1):
            states_dw = generate_states(n, ndw)
            dwn = states_dw.size
            d = dup * dwn
            eigvals = np.empty(d)
            add_onsites(ek, Uk, states_up, states_dw, eigvals, 0.)
            eigvals.sort()
            espace[(nup,ndw)] = Sector(
                States(states_up, states_dw),
                states_up.size*states_dw.size,
                states_up.size,
                states_dw.size,
                eigvals,
                None
            )
            egs = min(eigvals.min(), egs)
    return espace, egs


def screen_espace(espace, egs, beta=1e6, cutoff=1e-9):#, neig_sector=None, n=0):
    """Keep sectors containing relevant eigen-states:
    any{ exp( -beta * (E(N)-egs) ) } > cutoff. If beta
    is > ~1e3, then only the GS is kept.

    """

    delete = []
    for (nup, ndw), sct in espace.items():
        diff = np.exp(-beta*(sct.eigvals-egs)) > cutoff
        if diff.any():
            if (sct.eigvecs.ndim<2): 
                raise RuntimeError('sct.eigvecs.ndim < 2!')
            keep_idx = np.where(diff)[0]
            sct.eigvals = sct.eigvals[keep_idx]
            sct.eigvecs = sct.eigvecs[:,keep_idx]
            assert sct.eigvecs.ndim>1
        else:
            delete.append((nup,ndw))

    for k in delete:
        espace.pop(k)


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