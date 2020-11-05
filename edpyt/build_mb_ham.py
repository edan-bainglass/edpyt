import numpy as np
from numba import njit

# Sparse matrix format
from scipy.sparse import (
    csr_matrix
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
    c,
    cdg,
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
def add_hoppings(ix_s, states, H, c, vals, rows, cols, nzi, nzj):
    """Add hoppings from nonzero elements H[nzi, nzj]
    for state index ix_s in sparse matrix csr(vals, (rows, cols)).

    """
    s = states[ix_s]

    # if nzi is None:
    #     nzi, nzj = np.where(~np.eye(H.shape[0],dtype=bool))

    # c = 0
    # for i in nzi:
    #     for j in nzj:
    for i, j in zip(nzi, nzj):
        if ((s>>i)&unsiged_dt(1)) and (not (s>>j)&unsiged_dt(1)):
            # fsgn_t, t = c(s, j)
            t = flip(s, j)
            sgn_t = fsgn(s, j)
            f = flip(t, i)
            sgn_f = fsgn(t, i)
            # fsgn_f, sm = cdg(t, i)
            vals[c] += H[i, j] * sgn_t * sgn_f
            rows[c] += ix_s
            cols[c] += np.searchsorted(states, f)
            c += 1
            # if (not (s>>i)&unsiged_dt(1)) and ((s>>j)&unsiged_dt(1)):
            #     # fsgn_t, t = c(s, j)
            #     t = flip(s, i)
            #     sgn_t = fsgn(s, i)
            #     f = flip(t, j)
            #     sgn_f = fsgn(t, j)
            #     # fsgn_f, sm = cdg(t, i)
            #     vals[c] = H[j, i] * sgn_t * sgn_f
            #     cols[c] = ix_s
            #     rows[c] = np.searchsorted(states, f)
            #     c += 1

    print(c)
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

    i_vals = np.zeros(d, dtype=np.float64)

    # On-site terms :
    for i in range(d):

        iup, idw = get_spin_indices(i, dup, dwn)

        sup = states_up[iup]
        sdw = states_dw[idw]

        # Singly-occupied : \sum_{l,sigma} n^+_{l,sigma}
        onsite_energy = np.sum(H.diagonal() * binrep(sup^sdw, n))

        # Doubly-occupied : \sum_{l,sigma} n^+_{l,sigma} n_{l,sigma'}
        onsite_int = np.sum(V.diagonal() * binrep(sup&sdw, n))

        i_vals[i] = onsite_energy + onsite_int

    nzi, nzj = np.nonzero(H)
    neq = nzi != nzj
    nzi = nzi[neq]
    nzj = nzj[neq]
    data = H[nzi, nzj]
    H_coo_mat = coo_matrix((data, (nzi, nzj)), shape=(n,n))

    nnz_up_count = len(nzi) * int(binom(n-2, nup-1))
    zeros = lambda dtype: np.zeros(nnz_up_count, dtype=dtype)
    iup_coo_mat = coo_matrix(
        (zeros(np.float64), (zeros(np.uint32), zeros(np.uint32)))
        )
    iup_vals = np.zeros(nnz_up_count, dtype=np.float64)
    iup_rows = np.zeros(nnz_up_count, dtype=np.int32)
    iup_cols = np.zeros(nnz_up_count, dtype=np.int32)
    iup_coo_mat = coo_matrix((iup_vals))

    c = 0
    for iup in range(dup):
        c = add_hoppings(iup, states_up, H, c, iup_vals, iup_rows, iup_cols, nzi, nzj)

    iup_mat = csr_matrix((iup_vals, (iup_rows, iup_cols)), shape=(dup, dup))

    nnz_dw_count = len(nzi) * int(binom(n-2, ndw-1))
    idw_vals = np.zeros(nnz_dw_count, dtype=np.float64)
    idw_rows = np.zeros(nnz_dw_count, dtype=np.int32)
    idw_cols = np.zeros(nnz_dw_count, dtype=np.int32)

    c = 0
    for idw in range(dwn):
        c = add_hoppings(idw, states_dw, H, c, idw_vals, idw_rows, idw_cols, nzi, nzj)

    idw_mat = csr_matrix((idw_vals, (idw_rows, idw_cols)), shape=(dwn, dwn))

    return i_vals, iup_mat, idw_mat
