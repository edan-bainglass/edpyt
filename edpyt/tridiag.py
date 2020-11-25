import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

stebz = get_lapack_funcs('stebz', dtype=np.float64)
stein = get_lapack_funcs('stein', dtype=np.float64)
stemr = get_lapack_funcs('stemr', dtype=np.float64)
stemr_lwork = get_lapack_funcs('stemr_lwork', dtype=np.float64)


egs_tridiag = lambda a, b: stebz(a,b,2,0.,1.,1,1,0.,'E')[1][0]


def _solve_partial(a, b, select_range=(0,0), eigvals_only=False):
    il = select_range[0] + 1
    iu = select_range[1] + 1
    order = 'E' if eigvals_only else 'B'
    m, w, iblock, isplit, info = stebz(a,b,2,0.,1.,il,iu,0.,order)
    w = w[:m]
    if eigvals_only:
        return w, None
    v, info = stein(a,b,w,iblock,isplit)
    return w, v[:, :m]


def _solve_full(a, b, eigvals_only=False):
    b_ = np.empty(b.size+1, a.dtype)
    b_[:-1] = b
    compute_v = not eigvals_only
    lwork, liwork, info = stemr_lwork(a,b_,0,0.,1.,1,1,compute_v=compute_v)
    m, w, v, info = stemr(a,b_,0,0.,1.,1,1,compute_v=compute_v,lwork=lwork,liwork=liwork)
    return w, v


def eigh_tridiagonal(a, b, select=0, select_range=(0,0), eigvals_only=False):
    """Diagonalize tridiagonal matrix.

    Args:
        a : diagonal elements
        b : first off-diagonal elements
        select :
            - 0 full spectrum
            - 2 eigvals in index range
        select_range : if select == 2 take eigvals in index range.

    """
    assert select in [0,2], f"Invalid select type {select}."
    if select == 0:
        w, v = _solve_full(a, b, eigvals_only)
    if select == 2:
        w, v = _solve_partial(a, b, select_range, eigvals_only)
    if eigvals_only:
        return w
    return w, v
