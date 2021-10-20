import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
from pathlib import Path

path = Path(__file__).parent.absolute()
# print(path)
lib = ct.CDLL(path/'libfit.so')
fit = lib.fit
fit.restype = None
fit.argtypes = [ndpointer(ct.c_double),ct.POINTER(ct.c_int),
                ct.POINTER(ct.c_double),ct.c_int,ct.c_int,
                ndpointer(np.complex128),ct.c_double]


def fit_hybrid(nbath, nmats, delta, beta):
    
    x = np.empty(2*nbath, np.float64)
    delta = np.ascontiguousarray(delta, dtype=np.complex128)
    it = ct.c_int(0)
    fret = ct.c_double()
    fit(x, ct.byref(it), ct.byref(fret), ct.c_int(nbath), 
        ct.c_int(nmats), delta, ct.c_double(beta))
    return x


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    z = 1.j*(2*np.arange(3000)+1)*np.pi/70.
    f = 2*(z-np.sqrt(z**2-1))
    x = fit_hybrid(8, 3000, f, 70.)
    print(x)
    z = 1.j*(2*np.arange(300)+1)*np.pi/70.
    f = 2*(z-np.sqrt(z**2-1))
    delta = (x[None,8:]**2/(z[:,None]-x[None,:8])).sum(1)
    plt.plot(z.imag, f.real, 'r--', z.imag, f.imag, 'b--')
    plt.plot(z.imag, delta.real, 'r-o', z.imag, delta.imag, 'b-o')
    plt.savefig('fit.png',bbox_inches='tight')