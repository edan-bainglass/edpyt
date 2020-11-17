import numpy as np
from scipy.sparse.linalg.interface import LinearOperator
import _psparse

from scipy.sparse import (
    # kronsum(A_mm, B_nn) = kron(I_n,A) + kron(B,I_m)
    kronsum,
    # Sparse diagonal matrix
    diags
)

"""H = kron(I_dw,H_up) + kron(H_dw,I_up) + diag(H_dd).

"""


def matvec_operator(vec_diag, sp_mat_up, sp_mat_dw):
    """Sparse matrix vector operator.

    Returns:
        matvec : callable f(v)
            Returns returns H * v.

    """
    dup = sp_mat_up.shape[0]
    dwn = sp_mat_dw.shape[0]
    d = dup*dwn
    # sp_diag = sp_diag_matrix(vec_diag)
    # op = sp_diag + kronsum(sp_mat_up, sp_mat_dw)
    def matvec(vec):
        res = vec_diag * vec
        # res += sp_mat_up.dot(vec.reshape(dwn,dup).T).T.flatten()
        # res += sp_mat_dw.dot(vec.reshape(dwn,dup)).flatten()
        _psparse.UPmultiply(sp_mat_up,vec,res)
        _psparse.DWmultiply(sp_mat_dw,vec,res)
        return res
    return LinearOperator((d,d), matvec, vec_diag.dtype)


def todense(vec_diag, sp_mat_up, sp_mat_dw):
    """Construct dense matrix Hamiltonian explicitely.

    Returns:
        H

    """
    mat = diags(vec_diag) + kronsum(sp_mat_up, sp_mat_dw)
    return mat.todense()
