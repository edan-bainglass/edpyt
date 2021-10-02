import numpy as np

# Subclass for non-local operator
from scipy.sparse import csr_matrix

# Non-zero elements of sparse matrix lower than calculated.
from warnings import warn
from edpyt.ham_hopping import cs_type

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
def build_ham_non_local(Jx, Jp, states_up, states_dw, vec_diag):
    """
        Jx : direct (spin) exchange.
        Jp : pair (simultaneous) hopping of two electrons.
        Jh : coulomb assistend hopping.
    
    """
    #        __                      
    #       \              +   +                      +   +                     +                       
    #                 J   c   c    c     c     + J   c   c    c     c    +  J  c   c    ( n     + n   ) 
    #       /__        x    is  js'  is'  js      p    is  i-s  j-s  js      h   is  js     is'     js' 
    #       i!=j,
    #        ss'       
    dup = states_up.size
    dwn = states_dw.size
    n = Jx.shape[0]
    
    Nup = count_bits(states_up[0],n)
    Ndw = count_bits(states_dw[0],n)
    
    nnz_x = count_nnz_offdiag(Jx)
    nnz_p = count_nnz_offdiag(Jp)
    nnz = (nnz_x + nnz_p) * int(binom(n-2, Nup-1)) * int(binom(n-2, Ndw-1))
        
    sp_mat = empty_csrmat(nnz, (dwn*dup, dwn*dup))
    
    if np.any(Jx.diagonal()):
        warn('Neglecting off-diagonal exchange couplings.')
    if np.any(Jp.diagonal()):
        warn('Neglecting off-diagonal pair hopping couplings.')
    
    Jx = nnz_offdiag_csrmat(Jx, nnz_x)
    Jp = nnz_offdiag_csrmat(Jp, nnz_p)
    
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
            # Spin-Exchange: \sum_ij \sum_ss' 0.5 * Jx_ij [c^+_is c^+_js' c_is' c_js]
            #    s=s'  :: \sum_ij \sum_s - 0.5 * Jx_ij [c^+_is c_is] [c^+_js c_js]
            #    s!=s' :: \sum_ij - 0.5 * Jx_ij [S+_i S-_j + S+_j + S-_i]
            #          :: \sum_ij - Jx_ij [S+_i S-_j]
            tmp = 0.
            for i in range(n):
                if (not nup[i]) & ndw[i]:
                    for p in range(Jx.indptr[i], Jx.indptr[i+1]):
                        j = Jx.indices[p]
                        if nup[j] & (not ndw[j]):
                            sgn_up, fup = cdgc(sup, i, j)
                            jup = binsearch(states_up, fup)
                            sgn_dw, fdw = cdgc(sdw, j, i)
                            jdw = binsearch(states_dw, fdw)
                            Jndx = jdw * dup + jup
                            sp_mat.data[count] = sgn_up * sgn_dw * Jx.data[p]
                            sp_mat.indices[count] = Jndx                
                            count += 1
                for p in range(Jx.indptr[i], Jx.indptr[i+1]):
                    j = Jx.indices[p]
                    tmp += Jx.data[p] * (nup[i]*nup[j] + ndw[i]*ndw[j])             
            tmp *= - 0.5
            vec_diag[iup+idw*dup] += tmp
            # Pair-Hopping: \sum_ij \sum_s 0.5 * Jp_ij [c^+_is c^+_i-s c_j-s c_js]
            #   s'=-s  :: \sum_ij \sum_s 0.5 * Jp_ij [c^+_is c_js] [c^+_i-s c_j-s]
            #          :: \sum_ij Jp_ij [c^+_is c_js]
            for i in range(n):
                if (not nup[i]) & (not ndw[i]):
                    for p in range(Jp.indptr[i], Jp.indptr[i+1]):
                        j = Jp.indices[p]
                        if nup[j] & ndw[j]:
                            sgn_up, fup = cdgc(sup, i, j)
                            jup = binsearch(states_up, fup)
                            sgn_dw, fdw = cdgc(sdw, i, j)
                            jdw = binsearch(states_dw, fdw)
                            Jndx = jdw * dup + jup
                            sp_mat.data[count] = sgn_up * sgn_dw * Jp.data[p]
                            sp_mat.indices[count] = Jndx                
                            count += 1            
            sp_mat.indptr[Indx+1] = sp_mat.indptr[Indx] + (count-init_count)
    if count!=sp_mat.data.size:
        warn(f'Number of non-zero elements for sector ({Nup},{Ndw}) lower than calculated.')
        sp_mat = cs_type(sp_mat.data[:count], sp_mat.indptr, 
                         sp_mat.indices[:count], sp_mat.shape)
    sp_mat = NonLocal((sp_mat.data, sp_mat.indices, sp_mat.indptr),shape=sp_mat.shape)
    return sp_mat


class NonLocal(csr_matrix):
    """Non local Hamiltonian operator."""    
    def matvec(self, other, out):
        _psparse.Multiply(self, other, out)
        
    def todense(self, order=None, out=None):
        return np.asarray(super().todense(order=order, out=out))