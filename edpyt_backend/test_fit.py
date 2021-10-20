import numpy as np
import subprocess
from _fit import lib, ffi


def test_fit():
    
    nmats=3000
    nbath=8
    beta=70.

    z = 1.j*(2*np.arange(nmats)+1)*np.pi/beta
    z = z.astype(np.complex64)

    f = 2*(z-np.sqrt(z**2-1.))
    f_ptr = ffi.cast('float _Complex*', f.ctypes.data)

    x_ptr = lib.fit(nbath, nmats, f_ptr, beta)
    x = np.frombuffer(ffi.buffer(x_ptr, 2*nbath*4), np.float32)

    out = subprocess.check_output('./main').decode('utf-8').strip()
    out = np.fromstring(out, float, sep=' ')
    
    np.testing.assert_allclose(x, out, atol=1e-6)    

test_fit()
