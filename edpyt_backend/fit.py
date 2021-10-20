from pathlib import Path
from cffi import FFI

ffi = FFI()
cwd = Path().absolute()

ffi.cdef("void fit(double x[], int *iter, double *fret, \
         int nbath, int nmats, double _Complex *vals_true, double beta);")

ffi.set_source(
    "_fit",
    '#include "fit.h"',
    libraries=["m","fit"], # m is <math.h>, libfit.so is user
    library_dirs=[cwd.as_posix()],
    extra_link_args=['-Wl,-rpath='+cwd.as_posix()],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
