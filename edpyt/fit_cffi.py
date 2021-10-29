from edpyt_backend._fit import lib, ffi

_cfit = lib.fit

def fit_hybrid(x, nmats, delta, beta):
    nbath = x.size//2
    it = ffi.new('int*')
    fret = ffi.new('double* ')
    x_ptr = ffi.cast('double*', x.ctypes.data)
    delta_ptr = ffi.cast('double _Complex*', delta.ctypes.data)
    lib.fit(x_ptr, it, fret, nbath, nmats, delta_ptr, beta)
    return it[0], fret[0]


