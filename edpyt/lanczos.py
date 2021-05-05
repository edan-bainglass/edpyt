import numpy as np
from functools import partial
from numba import njit, prange
from scipy.linalg.blas import get_blas_funcs

from edpyt.tridiag import egs_tridiag, eigh_tridiagonal

axpy = get_blas_funcs('axpy', dtype=np.float64)
scal = get_blas_funcs('scal', dtype=np.float64)
swap = get_blas_funcs('swap', dtype=np.float64)


def sl_step(matvec, comm=None):
    """Simple Lanczos step.    
    
    Args:
        comm : if MPI communicator is given the hilbert space
            is assumed to be diveded along spin-down dimension.
    """
    if comm is None:
        return partial(_sl_step, matvec)
    else:
        return partial(_sl_step_mpi, matvec, comm=comm)


def _sl_step(matvec, v, l):
    """Simple Lanczos step.
    
    Given the current (\tilde(v)) and previous (l) Lanc. vectors,
    compute a single Lanczos step.

    Return:
        a : <v+1|v>
        b : <v|v>
        v : |v> / b
        v+1 : Av - av - bl

    """
    b = np.linalg.norm(v)
    scal(1/b,v)
    w = matvec(v)
    a = v.dot(w)
    # w -= (a[n] * v + b[n] * l)
    axpy(v,w,v.size,-a)
    axpy(l,w,l.size,-b)
    return a, b, v, w


def _sl_step_mpi(matvec, v, l, comm):
    """Same as sl_step, but with MPI support.

    Args:
        sl_lanc(*args[:-1])
        comm : MPI communicator
    """
    from mpi4py.MPI import SUM
    b2_local = v.dot(v)
    b = np.sqrt(comm.allreduce(b2_local, op=SUM))
    scal(1/b,v)
    w = matvec(v)
    a_local = v.dot(w)
    a = comm.allreduce(a_local, op=SUM)
    # w -= (a[n] * v + b[n] * l)
    axpy(v,w,v.size,-a)
    axpy(l,w,l.size,-b)    
    return a, b, v, w


def build_sl_tridiag(matvec, phi0, maxn=500, delta=1e-15, tol=1e-10, ND=10, comm=None):
    '''Build tridiagonal coeffs. with simple Lanczos method.

    Args:
        maxn : max. # of iterations
        delta : set threshold for min || b[n] ||.
        tol : set threshold for min change in groud state energy.
        ND : # of iterations to check change in groud state energy.
        comm : MPI communicator

    Returns:
        a : diagonal elements
        b : off-diagonal elements

    NOTE:
        1) T := diag(a,k=0) + diag(b[1:],k=1) + diag(b[1:],k=-1)
        2) with MPI support, the stopping condition is the same
        since both
            i) b[n]
            ii) groud state energy (coming from a[:n],b[:n]) 
        are known by all processes at each iteration.
    '''
    lanc_step = sl_step(matvec, comm)
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
            a[n], b[n], l, v = lanc_step(v, l)
            if (abs(b[n])<delta) or (n>=(maxn-1)):
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


@njit('(float64[:],float64[:],float64[:,:])',parallel=True,fastmath=True)
def _kron(u_row, l, r):
    """Helper function used in sl_solve."""
    for i in range(u_row.size):
        coeff = u_row[i]
        for k in prange(l.size):
            r[i,k] += coeff * l[k]


def sl_solve(matvec, a, b, v0=None, select=0, select_range=(0,0), eigvals_only=False, comm=None):
    """Lanczos second pass.

    The Lanczos projection is defined
        
        T = V^+ H V # tridiagonal matrix
    
    and T can be diagonalized
    
        U^+ T U = D
    
    Hence, the vectors in the original hilbert space are:
    
        U^+ V^+ H V U = D
        X = V U
    
    Example:
    
    import sympy
    V = sympy.MatrixSymbol('V',3,2)
    U = sympy.MatrixSymbol('U',2,2)
    X = V*U
    X.as_explicit()
    Matrix([
            [U[0, 0]*V[0, 0] + U[1, 0]*V[0, 1], U[0, 1]*V[0, 0] + U[1, 1]*V[0, 1]],
            [U[0, 0]*V[1, 0] + U[1, 0]*V[1, 1], U[0, 1]*V[1, 0] + U[1, 1]*V[1, 1]],
            [U[0, 0]*V[2, 0] + U[1, 0]*V[2, 1], U[0, 1]*V[2, 0] + U[1, 1]*V[2, 1]]])
    
    -->
    for i in range(U.shape[0]):
        X += V[:,i][:,None] U[i,:][None,:]
    <--

    """
    assert select in [0,2], f"Invalid select type {select}."
    w, U = eigh_tridiagonal(a, b[1:], select, select_range, eigvals_only)
    if eigvals_only:
        return w
    else:
        assert v0 is not None, f"Starting lanczos vector must be provided for eigenvectors."
    lanc_step = sl_step(matvec, comm)
    v = v0
    l = np.zeros_like(v)
    r = np.zeros((U.shape[1],v0.size),v0.dtype)
    for n in range(a.size):
        _, _, l, v = lanc_step(v, l)
        _kron(U[n],l,r)
    return w, r.T # use convention v[:,0] is a vector (here, in 'F' order)


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
