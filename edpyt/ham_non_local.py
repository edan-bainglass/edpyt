from types import MethodType
import numpy as np
from scipy.sparse import csr_matrix

from edpyt.shared import unsigned_one as uone
from edpyt.operators import cdgc
from edpyt.lookup import binsearch, count_bits
from edpyt.sector import binom
from edpyt.ham_hopping import (
    empty_csrmat, count_nnz_offdiag, 
    nnz_offdiag_csrmat)
from edpyt import _psparse

# @njit(cs_type(Array(float64, 2, 'C', readonly=True),int64))
def nnz_csrmat(A, nnz):
    """Compress elements of A in csr format.

    Args:
        A : matrix, must be readonly.
        nnz : # of nonzero elements.

    """
    n = np.int32(A.shape[0])
    sp_A = empty_csrmat(nnz, (n,n))
    count = 0
    for i in range(n):
        init_count = count
        for j in range(n):
            if abs(A[i,j]) > 1e-7:
                sp_A.indices[count] = j
                sp_A.data[count] = A[i,j]
                count += 1
        sp_A.indptr[i+1] = sp_A.indptr[i] + (count-init_count)
    return sp_A

# @njit((Array(float64, 3, 'A', readonly=True),
#        Array(float64, 2, 'C'),
#        Array(uint32, 1, 'C'),
#        Array(uint32, 1, 'C'),
#        Array(float64, 1, 'C'),
#        boolean,
#        float64),
#       parallel=True,cache=True)
def build_ham_non_local(Jx, Jp, states_up, states_dw):
    
    dup = states_up.size
    dwn = states_dw.size
    n = Jx.shape[0]
    
    nup = count_bits(states_up[0],n)
    ndw = count_bits(states_dw[0],n)
    
    nnz_offdiag = count_nnz_offdiag(Jx)
    nnz = np.count_nonzero(Jp)
    nnz_sp_count = (nnz_offdiag+nnz) * int(binom(n-2, nup-1)) * int(binom(n-2, ndw-1))
    sp_mat = empty_csrmat(nnz_sp_count, (dwn*dup, dwn*dup))
    
    Jx = nnz_offdiag_csrmat(Jx, nnz_offdiag)
    Jp = nnz_csrmat(Jp, nnz)
    
    count = 0
    for idw in range(dwn):
        nup = np.empty(n,np.uint32) # hoisted by numba
        ndw = np.empty(n,np.uint32) # hoisted by numba
        sdw = states_dw[idw]
        for i in range(n):
            ndw[i] = (sdw>>i)&uone 
        for iup in range(dup):
            init_count = count
            Indx = iup + dup * idw
            sup = states_up[iup]
            for i in range(n):
                nup[i] = (sup>>i)&uone
            # Spin-Exchange: Jx [c^+_j,dw c_i,dw] [c^+_i,up c_j,up]
            for i in range(n):
                if (~nup[i])&(ndw[i]):
                    for p in range(Jx.indptr[i], Jx.indptr[i+1]):
                        j = Jx.indices[p]
                        if (~ndw[j])&(nup[j])&(i!=j):
                            sgn_up, fup = cdgc(sup, i, j)
                            jup = binsearch(states_up, fup)
                            sgn_dw, fdw = cdgc(sdw, j, i)
                            jdw = binsearch(states_dw, fdw)
                            Jndx = jdw * dup + jup
                            sp_mat.data[count] = sgn_up * sgn_dw * Jx.data[p]
                            sp_mat.indices[count] = Jndx                
                            count += 1
            # Pair-Hopping: Jp [c^+_i,up c^+_i,dw] [c_j,dw c_j,up]
            for i in range(n):
                if (~nup[i])&(~ndw[i]):
                    for p in range(Jp.indptr[i], Jp.indptr[i+1]):
                        j = Jp.indices[p]
                        if (ndw[j])&(nup[j]):
                            sgn_up, fup = cdgc(sup, i, j)
                            jup = binsearch(states_up, fup)
                            sgn_dw, fdw = cdgc(sdw, i, j)
                            jdw = binsearch(states_dw, fdw)
                            Jndx = jdw * dup + jup
                            sp_mat.data[count] = sgn_up * sgn_dw * Jp.data[p]
                            sp_mat.indices[count] = Jndx                
                            count += 1            
            sp_mat.indptr[Indx+1] = sp_mat.indptr[Indx] + (count-init_count)
    sp_mat = NonLocal((sp_mat.data, sp_mat.indices, sp_mat.indptr),shape=sp_mat.shape)
    return sp_mat


class NonLocal(csr_matrix):
    """Non local Hamiltonian operator."""    
    def matvec(self, other, out):
        _psparse.Multiply(self, other, out)