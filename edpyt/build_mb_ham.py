import numpy as np
from numba import njit
from collections import namedtuple

# Sparse matrix format
from scipy.sparse import (
    csr_matrix
)

from sector import (
    generate_states,
    binom
)

from lookup import (
    binrep,
    get_spin_indices,
    count_bits
)

from shared import (
    unsiged_dt,
    params
)

from operators import (
    flip,
    fsgn,
    c,
    cdg
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


@njit(['UniTuple(int64,2)(int64,uint32[:],float64[:],int64[:],int64[:],int64,float64[:],int64[:])',
       'UniTuple(int64,2)(int64,uint32[:],float64[:],int64[:],int64[:],int64,float64[:],int32[:])'])
def add_hoppings(ix_s, states, T_data, T_r, T_c, cumcnt, val, col):
    """Add hoppings to many-body Hamiltonian.

    Args:
        ix_s : initial state (index) in `states`
        states : spin states
        T_data, T_r, T_r : hopping coo matrix vectors
        cumcnt : sequential index in many-body vectors
        val, col : many-body csr matrix vectors

    """
    s = states[ix_s]
    initcnt = cumcnt

    for i, j, elem in zip(T_r, T_c, T_data):
        if (not (s>>i)&unsiged_dt(1)) and ((s>>j)&unsiged_dt(1)):
            sgn_t, t = c(s, j)
            sgn_f, f = cdg(t, i)
            val[cumcnt] += elem * np.float64(sgn_t * sgn_f)
            col[cumcnt] += np.searchsorted(states, f)
            cumcnt += 1
    addcnt = cumcnt-initcnt
    return cumcnt, addcnt

@njit('float64(float64[:],uint32)')
def sum_diags_contrib(diags, s):
    """Sum diagonal energetic contributions for state s.

    Args:
        diags : onsite energies.
        s : state

    Return:
        res : sum of diagonal contrbutions.

    """
    n = diags.size
    res = 0.
    for i in range(n):
        if (s>>i)&unsiged_dt(1):
            res += diags[i]
    return res


def build_mb_ham(H, V, states_up, states_dw):
    """Build sparse Hamiltonian of the sector.

    Args:
        H : Hamiltonian (on-site + hopping) matrix (n x n).
        V : Interaction matrix (n x n).
        nup : number of up spins
        ndw : number of down spins
    """

    n = H.shape[0]
    ener_diags = H.diagonal().copy()
    int_diags = V.diagonal().copy()
    if abs(params['mu']) > 0.:
        mu = params['mu']
        ener_diags -= mu
    # Hartree term : \sum_{l,sigma} -0.5*U_l n^+_{l,sigma} + 0.25*U_l
    hfshift = 0.
    if params['hfmode'] == True:
        ener_diags -= 0.5 * int_diags
        hfshift = 0.25 * int_diags.sum()

    nup = count_bits(states_up[0],n)
    ndw = count_bits(states_dw[0],n)

    dup = states_up.size
    dwn = states_dw.size
    d = dup * dwn

    vec_diag = np.zeros(d, dtype=np.float64)

    # On-site terms :
    for i in range(d):

        onsite_energy = 0.
        onsite_int    = hfshift

        iup, idw = get_spin_indices(i, dup, dwn)

        sup = states_up[iup]
        sdw = states_dw[idw]

        # Singly-occupied : \sum_{l,sigma} n^+_{l,sigma}
        onsite_energy += sum_diags_contrib(ener_diags, sup)
        onsite_energy += sum_diags_contrib(ener_diags, sdw)

        # Doubly-occupied : \sum_{l,sigma} n^+_{l,sigma} n_{l,sigma'}
        onsite_int += sum_diags_contrib(int_diags, sup&sdw)

        vec_diag[i] = onsite_energy + onsite_int

    # Hoppings
    T = nnz_offdiag_coomat(H)
    nnz_offdiag = len(T.data)

    nnz_up_count = nnz_offdiag * int(binom(n-2, nup-1))
    sp_mat_up = empty_csrmat(nnz_up_count, (dup, dup))

    count = 0
    for iup in range(dup):
        count, added = add_hoppings(iup, states_up,
            # Hoppings
            T.data, T.row, T.col,
            # Sequential index in many-body nnz.
            count,
            # Many-Body Hamiltonian
            sp_mat_up.data, sp_mat_up.col)
        sp_mat_up.row[iup+1] = sp_mat_up.row[iup] + added

    nnz_dw_count = nnz_offdiag * int(binom(n-2, ndw-1))
    sp_mat_dw = empty_csrmat(nnz_dw_count, (dwn, dwn))

    count = 0
    for idw in range(dwn):
        count, added = add_hoppings(idw, states_dw,
            # Hoppings
            T.data, T.row, T.col,
            # Sequential index in many-body nnz.
            count,
            # Many-Body Hamiltonian
            sp_mat_dw.data, sp_mat_dw.col)
        sp_mat_dw.row[idw+1] = sp_mat_dw.row[idw] + added

    return (
        vec_diag,
        csr_matrix(
            (sp_mat_up.data, sp_mat_up.col, sp_mat_up.row),
            shape=sp_mat_up.shape),
        csr_matrix(
            (sp_mat_dw.data, sp_mat_dw.col, sp_mat_dw.row),
            shape=sp_mat_dw.shape)
    )


# scipy::sparse::coo_matrix might set index_type to np.int32.
_csr_matrix = namedtuple('csr_matrix',['data','row','col','shape'])


def nnz_offdiag_coomat(mat):
    """Return nonzero off-diagonal elements of matrix mat in
    scipy::sparse::coo_matrix format.

    """
    nzi, nzj = np.nonzero(mat)
    neq = nzi != nzj
    nzi = nzi[neq]
    nzj = nzj[neq]
    data = mat[nzi, nzj]
    mat = _csr_matrix(data, nzi, nzj, mat.shape)
    return mat

def empty_csrmat(nnz_count, shape):
    """Return scipy::sparse::coo_matrix with nnz_count elements.

    """
    zeros = lambda dtype: np.zeros(nnz_count, dtype=dtype)
    data = np.zeros(nnz_count, np.float64)
    row = np.zeros(shape[0]+1, np.int32)
    col = np.zeros(nnz_count, np.int32)
    mat = _csr_matrix(data, row, col, shape)
    return mat
