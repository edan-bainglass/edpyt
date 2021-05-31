# #cython: language_level=3
from numba import njit

import numpy as np

from edpyt.dcsrch_wrap import dcsrch

# dcsrch internal variables
FG = np.array([ord(i) for i in 'FG'], np.int8)
ERROR = np.array([ord(i) for i in 'ERROR'], np.int8)
WARN = np.array([ord(i) for i in 'WARN'], np.int8)
START = np.array([ord(i) for i in 'START                                                       '], np.int8)

@njit
def _minimize_bfgs(f, x0, maxiter=None, gtol=1e-5):

    if maxiter is None:
        maxiter = 200 * len(x0)

    old_fval = f(x0)
    gfk = approx_derivative(f, x0)

    k = 0
    N = len(x0)
    I = np.zeros((N,N), dtype=np.float64)
    for i in range(N):
        I[i,i] = 1.
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    warnflag = 0
    gnorm = np.amax(np.abs(gfk))
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        alpha_k, old_fval, old_old_fval, gfkp1 = \
                    _line_search_wolfe12(f, xk, pk, gfk,
                                        old_fval, old_old_fval, amin=1e-100, amax=1e100)
        if alpha_k is None:
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1
        assert gfkp1 is not None

        yk = gfkp1 - gfk
        gfk = gfkp1
        
        k += 1
        gnorm = np.amax(np.abs(gfk))
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        A1 = I - np.expand_dims(sk, -1) * np.expand_dims(yk, 0) * rhok
        A2 = I - np.expand_dims(yk, -1) * np.expand_dims(sk, 0) * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * np.expand_dims(sk, -1) * 
                                                 np.expand_dims(sk, 0))

    fval = old_fval

    return xk, fval

@njit(cache=True)
def _dense_difference(fun, x0, h):
    m = 1#f0.size
    n = x0.size
    J_transposed = np.empty((n, m))
    h_vecs = np.diag(h)

    for i in range(h.size):
        x1 = x0 - h_vecs[i]
        x2 = x0 + h_vecs[i]
        dx = x2[i] - x1[i]
        f1 = fun(x1)
        f2 = fun(x2)
        df = f2 - f1

        J_transposed[i] = df / dx

    if m == 1:
        J_transposed = np.ravel(J_transposed)

    return J_transposed.T


@njit(cache=True)
def approx_derivative(fun, x0):

    # by default we use rel_step
    EPS = np.finfo(np.float64).eps**(1/3)
    sign_x0 = (x0 >= 0).astype(np.float64) * 2 - 1
    h = EPS * sign_x0 * np.maximum(1.0, np.abs(x0))
    h = np.abs(h)
    
    return _dense_difference(fun, x0, h)

@njit
def scalar_search_wolfe1(f, phi0, old_phi0, derphi0, xk, pk,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    assert old_phi0 is not None
    if derphi0 != 0.:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.int32)
    dsave = np.zeros((13,), np.float64)
    task = START.copy()

    for _ in range(100):
        stp, phi1, derphi1, task = dcsrch(alpha1, phi1, derphi1,
                                          c1, c2, xtol, task,
                                          amin, amax, isave, dsave)
        if np.all(FG==task[:2]):
            alpha1 = stp
            phi1 = f(xk+stp*pk)
            gval = approx_derivative(f, xk+stp*pk)
            derphi1 = np.dot(gval, pk)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    if np.all(task[:5]==ERROR) or np.all(task[:4]==WARN):
        stp = None  # failed

    return stp, phi1, phi0, gval

@njit
def line_search_wolfe1(f, xk, pk, gfk, old_fval, old_old_fval, 
                       c1=1e-4, c2=0.9, 
                       amax=50, amin=1e-8, xtol=1e-14):

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval, gval = scalar_search_wolfe1(
            f, old_fval, old_old_fval, derphi0, xk, pk,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fval, old_fval, gval

@njit
def scalar_search_wolfe2(f, phi0, old_phi0, derphi0, xk, pk,
                         c1=1e-4, c2=0.9, amax=50, maxiter=10):
    assert amax is not None
    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    alpha1 = min(alpha1, amax)

    phi_a1 = f(xk+alpha1*pk)
    #derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    for i in range(maxiter):
        if alpha1 == 0 or alpha0 == amax:
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            break

        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star, gval = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, f,
                              phi0, derphi0, c1, c2, xk, pk)
            break
        
        gval = approx_derivative(f, xk+alpha1*pk)
        derphi_a1 = np.dot(gval,pk)

        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star, gval = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, f,
                              phi0, derphi0, c1, c2, xk, pk)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = f(xk+alpha1*pk)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None

    return alpha_star, phi_star, phi0, derphi_star, gval

@njit(cache=True)
def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    if denom < 1e-40:
        return None
    d1 = np.empty((2, 2))
    d1[0, 0] = dc ** 2
    d1[0, 1] = -db ** 2
    d1[1, 0] = -dc ** 3
    d1[1, 1] = db ** 3
    [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                    fc - fa - C * dc]).flatten())
    A /= denom
    B /= denom
    radical = B * B - 3 * A * C
    xmin = a + (-B + np.sqrt(radical)) / (3 * A)
    if not np.isfinite(xmin):
        return None
    return xmin

@njit(cache=True)
def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    db = b - a * 1.0
    if db < 1e-20:
        return None
    B = (fb - D - C * db) / (db * db)
    xmin = a - C / (2.0 * B)
    if not np.isfinite(xmin):
        return None
    return xmin

@njit
def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          f, phi0, derphi0, c1, c2, xk, pk):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    
    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            # with np.errstate(divide='raise', over='raise', invalid='raise'):
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            # with np.errstate(divide='raise', over='raise', invalid='raise'):
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = f(xk+a_j*pk)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:

            gval = approx_derivative(f, xk+a_j*pk)
            derphi_aj = np.dot(gval,pk)

            if abs(derphi_aj) <= -c2*derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            # val_star = None
            # valprime_star = None
            break
    return a_star, val_star, valprime_star, gval

@njit
def line_search_wolfe2(f, xk, pk, gfk, old_fval, old_old_fval, 
                       c1=1e-4, c2=0.9,
                       amax=None, maxiter=10):
    derphi0 = np.dot(gfk, pk)

    alpha_star, phi_star, old_fval, derphi_star, gval = scalar_search_wolfe2(
            f, old_fval, old_old_fval, derphi0, xk, pk, c1, c2, amax,
            maxiter=maxiter)

    return alpha_star, phi_star, old_fval, gval

@njit
def _line_search_wolfe12(f, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100):

    ret = line_search_wolfe1(f, xk, pk, gfk,
                             old_fval, old_old_fval, amin=amin, amax=amax)

    if ret[0] is None:
        # line search failed: try different one.
        ret = line_search_wolfe2(f, xk, pk, gfk,
                                    old_fval, old_old_fval, amax=amax)

    return ret


if __name__ == '__main__':

    from scipy.optimize import fmin_bfgs

    @njit
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2.5)**2
    
    xopt, eps = _minimize_bfgs(f, np.random.random(2))
    expected, eps = fmin_bfgs(f, np.random.random(2), full_output=True)[:2]

    np.testing.assert_allclose(xopt, expected)