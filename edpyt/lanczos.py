import numpy as np
from scipy.linalg import norm
from scipy.linalg.blas import get_blas_funcs

from edpyt.tridiag import (
    egs_tridiag
)

axpy = get_blas_funcs('axpy', dtype=np.float64)
scal = get_blas_funcs('scal', dtype=np.float64)
swap = get_blas_funcs('swap', dtype=np.float64)


def sl_step(matvec, v, l):
    """Given the current (\tilde(v)) and previous (l) Lanc. vectors,
    compute a single (simple) Lanc. step

    Return:
        a : <v+1|v>
        b : <v|v>
        v : |v> / b
        v+1 : Av - av - bl

    """
    b = norm(v)
    scal(1/b,v)
    w = matvec(v)
    a = v.dot(w)
    # w -= (a[n] * v + b[n] * l)
    axpy(v,w,v.size,-a)
    axpy(l,w,l.size,-b)
    return a, b, v, w


def build_sl_tridiag(matvec, phi0, maxn=500, delta=1e-15, tol=1e-10, ND=10):
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
            a[n], b[n], l, v = sl_step(matvec, v, l)
            if (abs(b[n])<delta) or (n>=maxn):
                converged = True
                break
            n += 1
        if n==1:
            converged = True
            break
        egs = egs_tridiag(a[:n],b[1:n])
        if abs(egs - egs_prev)<tol:
            converged = True
        else:
            egs_prev = egs

    return a[:n], b[:n]


def gram_schmidt_rows(X, row_vecs=True, norm = True):
    """Gram-Schmidt row major.

    """
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def gram_schmidt_columns(X):
    """Gram-Schmidt col major.

    """
    Q, R = np.linalg.qr(X)
    return Q


def build_bd_tridiag(matvec, phi0, maxn=300, delta=1e-15, tol=1e-10, ND=10):
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
    T = np.zeros((maxn,maxn), dtype=np.float64)
    # Loops vars.
    converged = False
    egs_prev = np.inf
    I = []
    vk = phi0 # Candidate Lanc. vectors. (1st is the current candidate)
    vl = [] # Previous n-Lc Lanc. vectors. (1st is the oldest)
    vd = [] # Deflated vectors. (1st is the oldest)
    #
    n = 0
    L = len(phi0)
    Lc = L
    while (not converged) and (n<(maxn-Lc)):
        v = vk[0]
        b = norm(v) # || v || _2
        if (abs(b)<delta): # Check deflation.
            if ((n-Lc)>=0):
                I.append(n-Lc)
                vd.append(vl[0])
            Lc -= 1
            if (Lc==0):
                converged = True
                break
            vk.pop(0)  # Eliminate v from candidates. k->k+1
            if len(vl)>Lc: # Eliminate v[n-Lc] from list of previous Lanc. vectors.
                vl.pop(0)
            continue # Go to b = norm(v[0]).
        T[n,n-Lc] = b # v become new Lanc. vector.
        v /= b
        vk.pop(0) # Eliminate v from candidates. k->k+1
        for k0, k in enumerate(range(n+1,n+Lc)): # Orthogonalize next candidates against v.
            T[n,k-Lc] = v.dot(vk[k0])
            vk[k0] -= T[n,k-Lc]*v
        w = matvec(v) # New candidate vector.
        for k0, k in enumerate(range(max(0,n-Lc),n)): # Orthogonalize against previous (n-Lc) Lanc. vectors.
            T[k,n] = np.conj(T[n,k])
            w -= T[k,n]*vl[k0]
        for k0, k in enumerate(I): # Orthogonalize against deflated vectors
            T[k,n] = vd[k0].dot(w)
            w -= vd[k0]*T[k,n]
        T[n,n] = v.dot(w) # Orthogonalize against current Lanc. vector
        w -= v*T[n,n]
        for k in I:
            T[n,k] = np.conj(T[k,n])
        if len(vl)==Lc:
            vl.pop(0)
        vl.append(v)
        vk.append(w)
        n += 1
        if ((n%ND)==0):
            egs = np.linalg.eigvalsh(T[:n,:n])[0]
            if abs(egs - egs_prev)<tol:
                converged = True
            else:
                egs_prev = egs

    return T[:n,:n]
