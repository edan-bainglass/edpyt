import numpy as np

from edpyt_backend._fit import lib, ffi

_cfit = lib.fit

def fit_hybrid(nbath, nmats, delta, beta):
    delta = np.ascontiguousarray(delta, dtype=np.complex64)
    delta_ptr = ffi.cast('float _Complex*', delta.ctypes.data)
    x = np.empty(2*nbath, dtype=np.float32)
    x_ptr = ffi.cast('float *', x.ctypes.data)
    iter = ffi.new("int *")
    fret = ffi.new("float *")
    _cfit(x_ptr, iter, fret, nbath, nmats, delta_ptr, beta)
    x = np.frombuffer(ffi.buffer(x_ptr, 2*nbath*4), np.float32)
    return x.astype(float)


