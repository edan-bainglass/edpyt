import numpy as np
from scipy.sparse.linalg.interface import LinearOperator
from edpyt import _psparse

from scipy.sparse import (
    # kronsum(A_mm, B_nn) = kron(I_n,A) + kron(B,I_m)
    kronsum,
    # Sparse diagonal matrix
    diags
)

"""H = kron(I_dw,H_up) + kron(H_dw,I_up) + diag(H_dd).

"""

def matvec_operator(vec_diag, sp_mat_up, sp_mat_dw, comm=None):
    """Sparse matrix vector operator.
    
    Args:
        comm : if MPI communicator is given the hilbert space
            is assumed to be diveded along spin-down dimension.
    """
    if comm is None:
        return _matvec_operator(vec_diag, sp_mat_up, sp_mat_dw)
    else:
        return _matvec_operator_mpi(vec_diag, sp_mat_up, sp_mat_dw, comm)


def _matvec_operator(vec_diag, sp_mat_up, sp_mat_dw):
    """Sparse matrix vector operator.

    Returns:
        matvec : callable f(v)
            Returns returns H * v.

    """
    def matvec(vec):
        res = vec_diag * vec
        _psparse.UPmultiply(sp_mat_up,vec,res)
        _psparse.DWmultiply(sp_mat_dw,vec,res)
        return res
    return matvec


def _matvec_operator_mpi(vec_diag, sp_mat_up, sp_mat_dw, comm):
    """Sparse matrix vector operator with MPI support.

    Returns:
        matvec : callable f(v)
            Returns returns H * v.

    """
    from edpyt.vector_transpose import collect_up, collect_dw
    size = comm.Get_size()
    dup = sp_mat_up.shape[0]
    dwn = sp_mat_dw.shape[0]
    # TODO : Implement general
    if dup!=dwn:
        raise NotImplementedError('Works only for sector with equal up and down # of spins.')
    if (dwn%comm.Get_size())!=0.:
        raise ValueError('# of MPI processes must be a divisor of hilbert up(& down) space.')
    dwn_local = dwn//size
    def matvec(vec):
        res = vec_diag * vec
        _psparse.UPmultiply(sp_mat_up,vec,res)
        vec = collect_dw(vec.reshape(dwn_local,dup),comm)
        res = collect_dw(res.reshape(dwn_local,dup),comm)
        _psparse.DWmultiply(sp_mat_dw,vec.reshape(-1,),res.reshape(-1,))
        vec = collect_up(vec,comm).reshape(-1,)
        res = collect_up(res,comm).reshape(-1,)
        return res
    return matvec


def todense(vec_diag, sp_mat_up=None, sp_mat_dw=None):
    """Construct dense matrix Hamiltonian explicitely.

    Returns:
        H

    """
    # old, much slower
    # mat = diags(vec_diag) + kronsum(sp_mat_up, sp_mat_dw)
    if sp_mat_up is None:
        return np.diag(vec_diag)
    dup = sp_mat_up.shape[0]
    dwn = sp_mat_dw.shape[0]
    mat = np.kron(np.eye(dwn), sp_mat_up.todense()) \
        + np.kron(sp_mat_dw.todense(), np.eye(dup))
    d = mat.shape[0]
    mat.flat[::d+1] += vec_diag
    return mat
