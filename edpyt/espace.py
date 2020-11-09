import numpy as np
from numba import njit
from scipy import linalg as la
from scipy.sparse import linalg as sla
from collections import defaultdict, namedtuple

from sector import (
    generate_states
)

from build_mb_ham import (
    build_mb_ham
)


from matvec_product import (
    todense,
    matvec_operator
)


States = namedtuple('States',['up','dw'])
Sector = namedtuple('Sector',['states','d','dup','dwn','eigvals','eigvecs'])


def solve_sector(H, V, states_up, states_dw, k=None):
    """Diagonalize sector.

    """
    if k:
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
    return sla.eigh(matvec, k, which='SA')


def build_espace(H, V):
    """Generate full spectrum.

    """
    n = H.shape[0]

    espace = defaultdict(Sector)

    # Groud state energy (GS)
    egs = np.inf
    for nup in range(n+1):
        states_up = generate_states(n, nup)
        for ndw in range(n+1):
            states_dw = generate_states(n, ndw)
            eigvals, eigvecs = solve_sector(H, V, states_up, states_dw)
            espace[(nup, ndw)] =  Sector(
                                       States(
                                           states_up, states_dw),
                                       states_up.size*states_dw.size,
                                       states_up.size,
                                       states_dw.size,
                                       eigvals,
                                       eigvecs)
            # Update GS energy
            egs = min(espace[(nup, ndw)].eigvals.min(), egs)

    return espace, egs
