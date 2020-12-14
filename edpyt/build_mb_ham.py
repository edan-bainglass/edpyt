import numpy as np
# Compiled
from numba import njit, prange
from numba.types import UniTuple, float64, int32, int64, uint32
from numba.experimental import jitclass

# Sparse matrix format
from scipy.sparse import (
    csr_matrix
)

from edpyt.sector import (
    generate_states,
    binom
)

from edpyt.lookup import (
    binsearch,
    get_spin_indices,
    count_bits
)

from edpyt.shared import (
    unsiged_dt,
    params
)

from edpyt.operators import (
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
spec = [
    ('shape',UniTuple(int32,2)),
    ('data',float64[:]),
    ('indptr',int32[:]),
    ('indices',int32[:]),
]

@jitclass(spec)
class cs:
    def __init__(self, data, indptr, indices, shape):
        self.shape = shape
        self.data = data
        self.indptr = indptr
        self.indices = indices

cs_type = cs.class_type.instance_type

@njit(cs_type(int32,UniTuple(int32,2)))
def empty_csrmat(nnz, shape):
    """Empty csr matrix.

    """
    data = np.empty(nnz,float64)
    indptr = np.empty(shape[0]+1,int32); indptr[0] = 0
    indices = np.empty(nnz,int32)
    return cs(data, indptr, indices, shape)


@njit(int64(float64[:,:]))
def count_nnz_offdiag(A):
    """Count the number of nonzeros outside the diagonal of A.

    """
    count = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if (i != j) and (abs(A[i,j]) > 1e-7):
                count += 1
    return count


@njit(cs_type(float64[:,:],int64))
def nnz_offdiag_csrmat(A, nnz):
    """Compress off-diagonal elements of A in csr format.

    Args:
        nnz : # of nonzero off-diagonal elements.

    """
    n = int32(A.shape[0])
    sp_A = empty_csrmat(nnz, (n,n))
    count = 0
    for i in range(n):
        init_count = count
        for j in range(n):
            if (i != j) and (abs(A[i,j]) > 1e-7):
                sp_A.indices[count] = j
                sp_A.data[count] = A[i,j]
                count += 1
        sp_A.indptr[i+1] = sp_A.indptr[i] + (count-init_count)
    return sp_A


@njit(int64(int64,uint32[:],cs_type,int64,cs_type))
def add_hoppings(ix_s, states, T, count, sp_mat):
    """Add hoppings to many-body Hamiltonian.

    Args:
        ix_s : initial state (index) in `states`.
        states : spin states.
        T: hopping matrix in csr format.
        count : sequential index in many-body vectors.
        sp_mat : many-body hamilton in csr format.

    """
    s = states[ix_s]
    init_count = count

    for i in range(T.shape[0]):
        # Check empty
        if ((s>>i)&unsiged_dt(1)):
            continue
        for p in range(T.indptr[i], T.indptr[i+1]):
            # Check occupied
            j = T.indices[p]
            if (not (s>>j)&unsiged_dt(1)):
                continue
            sgn_t, t = c(s, j)
            sgn_f, f = cdg(t, i)
            sp_mat.data[count] = T.data[p] * np.float64(sgn_t * sgn_f)
            sp_mat.indices[count] = binsearch(states, f)
            count += 1

    sp_mat.indptr[ix_s+1] = sp_mat.indptr[ix_s] + (count-init_count)
    return count


@njit('float64(float64[:],uint32)', parallel=False)
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
    for i in prange(n):
        if (s>>i)&unsiged_dt(1):
            res += diags[i]
    return res


@njit((float64[:],float64[:],uint32[:],uint32[:],float64[:],float64), parallel=False)
def add_onsites(ener_diags, int_diags, states_up, states_dw, vec_diag, hfshift):
    """On-site many-body hamiltonian.

    """
    dup = states_up.size
    dwn = states_dw.size

    # Temporary up contributions for performance.
    vec_diag_up = np.empty(dup)

    for iup in prange(dup):
        sup = states_up[iup]
        vec_diag_up[iup] = sum_diags_contrib(ener_diags, sup)

    for idw in prange(dwn):
        sdw = states_dw[idw]
        onsite_energy_dw = sum_diags_contrib(ener_diags, sdw)
        for iup in range(dup):
            sup = states_up[iup]
            i = iup + idw*dup
            onsite_int    = sum_diags_contrib(int_diags, sup&sdw)
            onsite_energy = vec_diag_up[iup] + onsite_energy_dw
            vec_diag[i] = onsite_energy + onsite_int + hfshift


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

    # Hartree term
    #  __
    # \                                U(l)             U(l)             U(l)
    #         U(l)  n(l,up) n(l,dw) -  ---  n(l,up)  -  ---  n(l,dw)  +  ---
    # /__ l                             2                2                4
    hfshift = 0.
    if params['hfmode'] == True:
        ener_diags -= 0.5 * int_diags
        hfshift = 0.25 * int_diags.sum()
        print('HF mode active')
        
    nup = count_bits(states_up[0],n)
    ndw = count_bits(states_dw[0],n)

    dup = states_up.size
    dwn = states_dw.size
    d = dup * dwn

    # On-site terms :
    vec_diag = np.empty(d)
    add_onsites(ener_diags, int_diags, states_up, states_dw, vec_diag, hfshift)

    # Hoppings
    nnz_offdiag = count_nnz_offdiag(H)
    T = nnz_offdiag_csrmat(H, nnz_offdiag)

    nnz_up_count = nnz_offdiag * int(binom(n-2, nup-1))
    sp_mat_up = empty_csrmat(nnz_up_count, (dup, dup))

    count = 0
    for iup in range(dup):
        count = add_hoppings(iup, states_up,
            # Hoppings
            T,
            # Sequential index in many-body nnz.
            count,
            # Many-Body Hamiltonian
            sp_mat_up)

    nnz_dw_count = nnz_offdiag * int(binom(n-2, ndw-1))
    sp_mat_dw = empty_csrmat(nnz_dw_count, (dwn, dwn))

    count = 0
    for idw in range(dwn):
        count = add_hoppings(idw, states_dw,
            # Hoppings
            T,
            # Sequential index in many-body nnz.
            count,
            # Many-Body Hamiltonian
            sp_mat_dw)

    return (
        vec_diag,
        csr_matrix(
            (sp_mat_up.data, sp_mat_up.indices, sp_mat_up.indptr),
            shape=sp_mat_up.shape),
        csr_matrix(
            (sp_mat_dw.data, sp_mat_dw.indices, sp_mat_dw.indptr),
            shape=sp_mat_dw.shape)
    )
