import numpy as np
from numba import njit
from scipy import linalg as la
from scipy.sparse import linalg as sla
from collections import namedtuple, defaultdict

from sector import (
    generate_states,
    get_sector_dim
)

from build_mb_ham import (
    build_mb_ham
)


from matvec_product import (
    todense,
    matvec_operator
)

from lookup import (
    get_sector_index
)


States = namedtuple('States',['up','dw'])
Sector = namedtuple('Sector',['states','d','dup','dwn','eigvals','eigvecs'])


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
    if k < (states_up.size*states_dw.size):
        eigvals, eigvecs = _solve_arpack(H, V, states_up, states_dw, k)
    else:
        eigvals, eigvecs = _solve_lapack(H, V, states_up, states_dw)
    return eigvals, eigvecs


def _solve_lapack(H, V, states_up, states_dw):
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
    return sla.eigsh(matvec, k, which='SA')


def build_espace(H, V, neig_sector=None, cutoff=np.inf):
    """Generate full spectrum.

    Args:
        neig_sector : # of eigen states in each sector.
        cutoff : discard energies for which exp^(-beta(e-egs)) < cutoff.

    Return:
        eig_space : list of eigen states ordered by energy.

    Return:
        eig_space
    """
    n = H.shape[0]

    espace = defaultdict(Sector)
    if neig_sector is None:
        neig_sector = np.zeros((n+1)*(n+1),int)
        for nup, ndw in np.ndindex(n+1,n+1):
            neig_sector[
                get_sector_index(nup,ndw,n)
            ] = get_sector_dim(n,nup)*get_sector_dim(n,ndw)
    # es_vals = []
    # es_vecs = []
    # es_isct = []
    #
    # # Helper function
    # def insert(idx, w, v, isct):
    #     es_vals.insert(idx, w)
    #     es_vecs.insert(idx, v)
    #     es_isct.insert(idx, isct)

    # Fill in eigen states in eigen space
    egs = np.inf
    for nup in range(n+1):
        states_up = generate_states(n, nup)
        for ndw in range(n+1):
            # Sequential index sector.
            isct = get_sector_index(nup, ndw, n)
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
            # if nup < ndw:
            #     espace[(ndw,nup)] = Sector(
            #         States(states_dw, states_up),
            #         states_up.size*states_dw.size,
            #         states_dw.size,
            #         states_up.size,
            #         eigvals,
            #         eigvecs
            #     )

            # Update GS energy
            egs = min(eigvals.min(), egs)

            # for i in range(eigvals.size):
            #     w = eigvals[i]
            #     v = eigvecs[:,i]
            #     idx = np.searchsorted(eig_space[0], w)
            #     # Update sector (nup, ndw).
            #     insert(idx, w, v, isct)
            #     # Update degenerate sector (ndw, nup).
            #     insert(idx+1, w, v, get_sector_index(ndw,nup,n))


    # # GS energy
    # egs = es_vals[0]

    return espace, egs
