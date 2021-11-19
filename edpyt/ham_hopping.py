from collections import namedtuple
from types import MethodType
import numpy as np

from numba import njit, prange
from numba.types import int64, uint32

from edpyt.lookup import binsearch, count_bits
from edpyt.operators import (cdgc, check_empty, check_full)
from edpyt.sector import binom
from scipy.sparse import csr_matrix
from edpyt import _psparse

# from numba.types import UniTuple, float64, int32, int64, uint32, Array

# spec = [
#     ('shape',UniTuple(int32,2)),
#     ('data',float64[:]),
#     ('indptr',int32[:]),
#     ('indices',int32[:]),
# ]

# @jitclass(spec)
# class cs:
#     def __init__(self, data, indptr, indices, shape):
#         self.shape = shape
#         self.data = data
#         self.indptr = indptr
#         self.indices = indices

# cs_type = cs.class_type.instance_type

cs_type = namedtuple('cs_type',['data','indptr','indices','shape'])

# @njit(cs_type(int32,UniTuple(int32,2)))
def empty_csrmat(nnz, shape):
    """Empty csr matrix.

    """
    data = np.empty(nnz,np.float64)
    indptr = np.empty(shape[0]+1,np.int32); indptr[0] = 0
    indices = np.empty(nnz,np.int32)
    return cs_type(data, indptr, indices, shape)

@njit#(int64(Array(float64, 2, 'C', readonly=True)))
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


# @njit#(cs_type(Array(float64, 2, 'C', readonly=True),int64))
def nnz_offdiag_csrmat(A, nnz):
    """Compress off-diagonal elements of A in csr format.

    Args:
        A : matrix, must be readonly.
        nnz : # of nonzero off-diagonal elements.

    """
    n = np.int32(A.shape[0])
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


# @njit#(int64(int64,uint32[:],cs_type,int64,cs_type))
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
        if check_empty(s, i):
            for p in range(T.indptr[i], T.indptr[i+1]):
                # Check occupied
                j = T.indices[p]
                if check_full(s, j):
                    sgn, f = cdgc(s, i, j)
                    sp_mat.data[count] = T.data[p] * np.float64(sgn)
                    sp_mat.indices[count] = binsearch(states, f)
                    count += 1

    sp_mat.indptr[ix_s+1] = sp_mat.indptr[ix_s] + (count-init_count)
    return count


def build_ham_hopping(H, sct):
    
    if not hasattr(sct.states, 'up'):
        raise NotImplementedError
    
    states_up = sct.states.up
    states_dw = sct.states.dw
    
    n = H.shape[-1]
    nup = count_bits(states_up[0],n)
    ndw = count_bits(states_dw[0],n)
    
    dup = states_up.size
    dwn = states_dw.size
    
    # Hoppings UP
    nnz_offdiag = count_nnz_offdiag(H[0])
    T = nnz_offdiag_csrmat(H[0], nnz_offdiag)

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

    # Hoppings DW
    nnz_offdiag = count_nnz_offdiag(H[1])
    T = nnz_offdiag_csrmat(H[1], nnz_offdiag)
        
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
    
    sp_mat_up = UpHopping((sp_mat_up.data, sp_mat_up.indices, sp_mat_up.indptr),dwn,shape=sp_mat_up.shape)
    sp_mat_dw = DwHopping((sp_mat_dw.data, sp_mat_dw.indices, sp_mat_dw.indptr),dup,shape=sp_mat_dw.shape)
    return sp_mat_up, sp_mat_dw

class UpHopping(csr_matrix):
    """Up hopping operator."""  
    def __init__(self, arg1, dwn, shape=None, dtype=None, copy=False):
        self.dwn = dwn
        super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)  
        
    def todense(self):
        return np.kron(np.eye(self.dwn), np.asarray(super().todense()))
    
    def matvec(self, other, out):
        _psparse.UPmultiply(self, other, out)
        

class DwHopping(csr_matrix):
    """Down hopping operator."""
    def __init__(self, arg1, dup, shape=None, dtype=None, copy=False):
        self.dup = dup
        super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)  
    
    def todense(self):
        return np.kron(np.asarray(super().todense()), np.eye(self.dup))
    
    def matvec(self, other, out):
        _psparse.DWmultiply(self, other, out)