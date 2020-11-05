import numpy as np
from numba import njit

# Sparse matrix format
from scipy.sparse import (
    coo_matrix
)

from sector import (
    generate_states,
    binom
)

from lookup import (
    binrep,
    get_spin_indices
)

from shared import (
    unsiged_dt
)

from operators import (
    flip,
    fsgn
)


"""
Conventions:
    n : number of levels per spin
    nup : number of up spins
    ndw : number of dw spins
    d : sector size
    dup : sector number of up spin states (all possible permutations of nup in n)
    dwn : sectot number of down spin states (all possible permutations of ndw in n)
"""


@njit
def add_hoppings(ix_s, states, T_data, T_r, T_c, c, val, row, col):
    """Add hoppings to many-body Hamiltonian.

    Args:
        ix_s : initial state (index) in `states`
        states : spin states
        T_data, T_r, T_r : hopping coo matrix vectors
        c : sequential index in many-body vectors
        val, row, col : many-body coo matrix vectors

    """
    s = states[ix_s]

    for i, j, elem in zip(T_r, T_c, T_data):
        if ((s>>i)&unsiged_dt(1)) and (not (s>>j)&unsiged_dt(1)):
            t = flip(s, j)
            sgn_t = fsgn(s, j)
            f = flip(t, i)
            sgn_f = fsgn(t, i)
            val[c] += elem * sgn_t * sgn_f
            row[c] += ix_s
            col[c] += np.searchsorted(states, f)
            c += 1
    return c


def build_mb_ham(H, V, nup, ndw):
    """Build sparse Hamiltonian of the sector.

    Args:
        H : Hamiltonian (on-site + hopping) matrix (n x n).
        V : Interaction matrix (n x n).
        nup : number of up spins
        ndw : number of down spins
    """

    n = H.shape[0]

    states_up = generate_states(n, nup)
    states_dw = generate_states(n, ndw)

    dup = states_up.size
    dwn = states_dw.size
    d = dup * dwn

    vec_diag = np.zeros(d, dtype=np.float64)

    # On-site terms :
    for i in range(d):

        iup, idw = get_spin_indices(i, dup, dwn)

        sup = states_up[iup]
        sdw = states_dw[idw]

        # Singly-occupied : \sum_{l,sigma} n^+_{l,sigma}
        onsite_energy = np.sum(H.diagonal() * binrep(sup, n))
        onsite_energy += np.sum(H.diagonal() * binrep(sdw, n))

        # Doubly-occupied : \sum_{l,sigma} n^+_{l,sigma} n_{l,sigma'}
        onsite_int = np.sum(V.diagonal() * binrep(sup&sdw, n))

        vec_diag[i] = onsite_energy + onsite_int

    # Hoppings
    T = nnz_offdiag_coomat(H)
    nnz_offdiag = len(T.data)

    nnz_up_count = nnz_offdiag * int(binom(n-2, nup-1))
    sp_mat_up = empty_coomat(nnz_up_count, (dup, dup))

    c = 0
    for iup in range(dup):
        c = add_hoppings(iup, states_up,
            # Hoppings
            T.data, T.row, T.col,
            # Sequential index in many-body nnz.
            c,
            # Many-Body Hamiltonian
            sp_mat_up.data, sp_mat_up.row, sp_mat_up.col)

    nnz_dw_count = nnz_offdiag * int(binom(n-2, ndw-1))
    sp_mat_dw = empty_coomat(nnz_dw_count, (dwn, dwn))

    c = 0
    for idw in range(dwn):
        c = add_hoppings(idw, states_dw,
            # Hoppings
            T.data, T.row, T.col,
            # Sequential index in many-body nnz.
            c,
            # Many-Body Hamiltonian
            sp_mat_dw.data, sp_mat_dw.row, sp_mat_dw.col)

    return vec_diag, sp_mat_up, sp_mat_dw



def nnz_offdiag_coomat(mat):
    """Return nonzero off-diagonal elements of matrix mat in
    scipy::sparse::coo_matrix format.

    """
    nzi, nzj = np.nonzero(mat)
    neq = nzi != nzj
    nzi = nzi[neq]
    nzj = nzj[neq]
    data = mat[nzi, nzj]
    return coo_matrix((data, (nzi, nzj)), shape=mat.shape)


def empty_coomat(nnz_count, shape, data_dt=np.float64):
    """Return scipy::sparse::coo_matrix with nnz_count elements.

    """
    zeros = lambda dtype: np.zeros(nnz_count, dtype=dtype)
    rows = zeros(unsiged_dt)
    cols = zeros(unsiged_dt)
    data = zeros(data_dt)
    return coo_matrix((data, (rows, cols)), shape=shape)
