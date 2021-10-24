import numpy as np
from scipy.optimize import fmin_cg
from numba import guvectorize, njit


"""Look at DCore github for example codes."""
   

@guvectorize('(float64[:],complex128[:],complex128[:])', '(n),(m)->(m)')
def _delta(p, z, out):
    ek = p[:p.size//2]
    vk2 = p[p.size//2:]**2
    for i in range(z.size):
        out[i] = 0.
        for j in range(ek.size):
            out[i] += vk2[j] / (z[i]-ek[j])


def Delta(z):
    out = np.empty(z.size, complex)
    def inner(p):
        return _delta(p, z, out)
    return inner
        

@guvectorize('(float64[:],complex128[:],complex128[:,:])', '(n),(m)->(m,n)')
def _ddelta(p, z, out):
    ek = p[:p.size//2]
    vk = p[p.size//2:]
    vk2 = vk**2
    for i in range(z.size):
        for j in range(ek.size):
            out[i,j] = vk2[j] / (z[i]-ek[j])**2
            out[i,j+ek.size] = 2*vk[j] / (z[i]-ek[j])


def DDelta(z, n):
    out = np.empty((z.size, n), complex)
    def inner(p):
        return _ddelta(p, z, out)
    return inner


@njit('float64(complex128[:],complex128[:])')
def _chi2(x, xt):
    res = 0.
    for i in range(x.size):
        res += ((x[i].real-xt[i].real)**2 + (x[i].imag-xt[i].imag)**2)
    return res / x.size


def Chi2(delta, vals_true):
    def inner(p):
        return _chi2(vals_true, delta(p))
    return inner


def dChi2(delta, ddelta, vals_true):
    F = np.empty_like(vals_true)
    def inner(p):
        np.subtract(vals_true, delta(p), out=F)
        dx = ddelta(p)
        return -2. * (F.real.dot(dx.real) + F.imag.dot(dx.imag)) / vals_true.size
    return inner
 

def fit_hybrid(p, nmats, vals_true, beta):
    z = 1.j*(2*np.arange(nmats)+1)*np.pi/beta
    delta = Delta(z)
    ddelta = DDelta(z, p.size)
    chi2 = Chi2(delta, vals_true)
    dchi2 = dChi2(delta, ddelta, vals_true)
    p[:] = fmin_cg(chi2, p, dchi2, disp=False)


def get_initial_bath(*, p=None, nbath=None, bandwidth=2.):
    """
    Args:
        nbath : # of bath sites.
        bandwidth : half-bandwidth for the bath initialization.
    """
    if p is None:
        p = np.empty(2 * nbath)
    elif nbath is None:
        nbath = p.size // 2
    else:
        raise ValueError("Must provide either nbath or initial parameters.")
    ek = p[:nbath]
    vk = p[nbath:]
    # Hoppings
    vk[:] = max(0.1, 1/np.sqrt(nbath))
    # Energies
    # ODD  : [-2,-1,0,+1,+2]
    # EVEN : [-2,-1,-0.1,0.1,+1,+2]
    ek[0] = -bandwidth
    ek[-1] = bandwidth
    nhalf = nbath//2
    nbath_is_odd = bool(nbath&1)
    nbath_is_even = not nbath_is_odd
    if nbath_is_even and nbath>=4:
        de = bandwidth/max(nhalf-1,1)
        ek[nhalf-1] = -0.1    
        ek[nhalf] = 0.1    
        for i in range(1,nhalf-1):
            ek[i] = -bandwidth + i*de
            ek[nbath-i-1] = bandwidth - i*de
    if nbath_is_odd and nbath>=3:
        de = bandwidth/nhalf
        ek[nhalf] = 0.
        for i in range(1,nhalf):
            ek[i] = -bandwidth + i*de
            ek[nbath-i-1] = bandwidth - i*de
    return p    

   
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import logging
    import time
    logging.basicConfig(filename='timing_fit_cg.txt', filemode='a', level='INFO')
    z = 1.j*(2*np.arange(3000)+1)*np.pi/70.
    f = 2*(z-np.sqrt(z**2-1))
    # Init
    x = get_initial_bath(nbath=8)
    fit_hybrid(x, 3000, f, 70.)
    # Timing
    start = time.perf_counter()
    for _ in range(10):
        x = get_initial_bath(nbath=8)
        fit_hybrid(x, 3000, f, 70.)
    elapsed = time.perf_counter() - start
    logging.info(f"Function took : {elapsed}")
    # Testing
    expect = np.array([-1.98663495, -1.28445211, -0.42108352, -0.05721278,  0.05721278,  0.42108352,
                        1.28445211,  1.98663495,  0.08997913,  0.18150846,  0.59537434,  0.31923968,
                        0.31923968,  0.59537434,  0.18150846,  0.08997913])
    np.testing.assert_allclose(expect, x)
    # Plot
    z = 1.j*(2*np.arange(300)+1)*np.pi/70.
    f = 2*(z-np.sqrt(z**2-1))
    delta = (x[None,8:]**2/(z[:,None]-x[None,:8])).sum(1)
    plt.plot(z.imag, f.real, 'r--', z.imag, f.imag, 'b--')
    plt.plot(z.imag, delta.real, 'r-o', z.imag, delta.imag, 'b-o')
    plt.savefig('fit.png',bbox_inches='tight')