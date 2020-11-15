import numpy as np
from scipy.linalg import norm
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs


axpy = get_blas_funcs('axpy', dtype=np.float64)
scal = get_blas_funcs('scal', dtype=np.float64)
swap = get_blas_funcs('swap', dtype=np.float64)
stebz = get_lapack_funcs('stebz', dtype=np.float64)


egs_tridiag = lambda a, b: stebz(a,b,2,0.,1.,1,1,0.,'E')[1][0]


def build_sl_tridiag(matvec, phi0, maxn=300, delta=1e-15, tol=1e-10, ND=10):
    '''Build tridiagonal coeffs. with simple Lanczos method.

    Args:
        maxn : max. # of iterations
        delta : set threshold for min || b[n] ||.
        tol : set threshold for min change in groud state energy.
        ND : # of iterations to check change in groud state energy.

    Returns:
        a : diagonal elements
        b : off-diagonal elements

    Note:
        T := diag(a,k=0) + diag(b[1:],k=1) + diag(b[1:],k=-1)

    '''
    a = np.empty(maxn, dtype=np.float64)
    b = np.empty(maxn, dtype=np.float64)
    # Loops vars.
    converged = False
    egs_prev = np.inf
    l = np.zeros_like(phi0)
    v = phi0
    #
    n = 0
    while not converged:
        for _ in range(ND):
            b[n] = norm(v)
            if (abs(b[n])<delta) or (n>=maxn):
                converged = True
                break
            scal(1/b[n],v)
            w = matvec(v)
            a[n] = v.dot(w)
            # w -= a[n] * v + b[n] * l
            axpy(v,w,v.size,-a[n])
            axpy(l,w,l.size,-b[n])
            l = v
            v = w
            n += 1
        egs = egs_tridiag(a[:n],b[1:n])
        if abs(egs - egs_prev)<tol:
            converged = True
        else:
            egs_prev = egs

    return a[:n], b[:n]
