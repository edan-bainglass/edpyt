#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse
from libc.stddef cimport ptrdiff_t
from cython.parallel import parallel, prange

#-----------------------------------------------------------------------------
# Headers
#-----------------------------------------------------------------------------

ctypedef int csi

ctypedef struct cs:
    # matrix in compressed-column or triplet form
    csi nzmax       # maximum number of entries
    csi m           # number of rows
    csi n           # number of columns
    csi *p          # column pointers (size n+1) or col indices (size nzmax)
    csi *i          # row indices, size nzmax
    double *x       # numerical values, size nzmax
    csi nz          # # of entries in triplet matrix, -1 for compressed-col

cdef extern csi csr_gaxpy (cs *A, double *x, double *y) nogil
cdef extern csi csr_saxpy (cs *A, double *x, double *y, csi i, csi n) nogil

assert sizeof(csi) == 4

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------


@cython.boundscheck(False)
def UPmultiply(X not None, np.ndarray[ndim=1, mode='c', dtype=np.float64_t] W not None,
              np.ndarray[ndim=1, mode='c', dtype=np.float64_t] result not None):
    """Multiply a UP spin.

    """
    if X.format == 'csc':
        raise NotImplementedError('csc format not supported.')

    cdef int i, nup, ndw
    cdef cs csX
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indptr  = X.indptr
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indices = X.indices
    cdef np.ndarray[double, ndim=1, mode = 'c'] data = X.data

    csX.nzmax = X.data.shape[0]
    csX.m = X.shape[0]
    csX.n = X.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = 1

    nup = X.shape[0]
    ndw = W.size // nup
    for i in prange(ndw, nogil=True):
        # Parallelize over rows.
        csr_gaxpy(&csX, &W[i*nup], &result[i*nup])


@cython.boundscheck(False)
def DWmultiply(X not None, np.ndarray[ndim=1, mode='c', dtype=np.float64_t] W not None,
           np.ndarray[ndim=1, mode='c', dtype=np.float64_t] result not None):
    """Multiply DW spin component.

    """
    if X.format == 'csc':
        raise NotImplementedError('csc format not supported.')

    cdef int i, nup, ndw
    cdef cs csX
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indptr  = X.indptr
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indices = X.indices
    cdef np.ndarray[double, ndim=1, mode = 'c'] data = X.data

    # Pack the scipy data into the CSparse struct. This is just copying some
    # pointers.
    csX.nzmax = X.data.shape[0]
    csX.m = X.shape[0]
    csX.n = X.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = 1

    ndw = X.shape[0]
    nup = W.size // ndw
    for i in prange(ndw, nogil=True):
        # Parallelize over rows
        csr_saxpy(&csX, &W[0], &result[0], i, nup)


@cython.boundscheck(False)
def Multiply(X not None, np.ndarray[ndim=1, mode='c', dtype=np.float64_t] W not None,
           np.ndarray[ndim=1, mode='c', dtype=np.float64_t] result not None):
    """Multiply full vector.

    """
    if X.format == 'csc':
        raise NotImplementedError('csc format not supported.')

    cdef int i, nup, ndw, p
    cdef cs csX
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indptr  = X.indptr
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indices = X.indices
    cdef np.ndarray[double, ndim=1, mode = 'c'] data = X.data

    # Pack the scipy data into the CSparse struct. This is just copying some
    # pointers.
    csX.nzmax = X.data.shape[0]
    csX.m = X.shape[0]
    csX.n = X.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = 1

    for i in prange(csX.m, nogil=True):
        # Parallelize over rows
        for p in range(csX.p[i], csX.p[i+1]):
            result[i] += csX.x[p] * W[ csX.i[p] ]