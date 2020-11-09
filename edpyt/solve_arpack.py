import numpy as np

# Use ARPACK
from scipy.sparse.linalg import eigh

from matvec_product import (
    matvec_operator
)


states_ = namedtuple('states',['up','dw'])
sector_ = namedtuple('sector',['states','d','dup','dwn','eigvals','eigvecs'])


def solve_sector(H, V, nup, ndw, k=6):
    """Diagonalize sector with ARPACK.

    """
    n = H.shape[0]

    # Groud state energy (GS)
    states_up = generate_states(n, nup)
    states_dw = generate_states(n, ndw)
    matvec = matvec_operator(
        *build_mb_ham(H, V, states_up, states_dw)
    )
    eigvals, eigvecs = eigh(matvec, k, which='SA')
    return sector_(
                states_(
                    states_up, states_dw),
                states_up.size*states_dw.size,
                states_up.size,
                states_dw.size,
                eigvals,
                eigvecs)

# sectkeys = np.indices((n+1,n+1)).reshape(2,-1).T
# getsector = lambda nup, ndw: np.ravel_multi_index((nup,ndw),(n+1,n+1))
# getnspins = lambda isector: np.unravel_index(isector, (n+1,n+1))

def build_gf_lanczos(nup, dwn):
    pass
