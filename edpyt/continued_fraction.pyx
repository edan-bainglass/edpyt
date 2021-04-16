cimport cython
from cython.parallel import parallel, prange

import numpy as np
# this eneables Cython enhanced compatibilities
cimport numpy as np

cdef complex calc(complex z,
                  Py_ssize_t n,
                  complex *a,
                  complex *b) nogil:

    
    cdef complex r = 0.
    cdef Py_ssize_t i

    for i in range(n-1,-1,-1):
        r = b[i] / (z-a[i]-r)

    return r


def continued_fraction(z,
                       np.ndarray[double, ndim=1, mode='c'] a,
                       np.ndarray[double, ndim=1, mode='c'] b):

    cdef np.ndarray[complex, ndim=1, mode='c'] _z = np.atleast_1d(z)
    cdef np.ndarray[complex, ndim=1, mode='c'] _a = a.astype(complex)
    cdef np.ndarray[complex, ndim=1, mode='c'] _b = b.astype(complex)
    cdef Py_ssize_t i, m = _z.size, n = a.size
    cdef np.ndarray[complex, ndim=1, mode='c'] out = np.empty(m, complex) 

    for i in prange(m, nogil=True):
        out[i] = calc(_z[i], n, &_a[0], &_b[0])

    return np.squeeze(out)