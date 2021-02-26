import numpy as np

from edpyt.espace import build_espace, screen_espace
from edpyt.dedlib import (get_evecs_occupation, get_occupation)


t = 1.
U = 1.
ed = 0.5

H = np.array([
    [ed,-t],
    [-t,ed]
])

V = np.array([
    [U,0],
    [0,U]
])

# https://www.cond-mat.de/events/correl20/manuscripts/pavarini.pdf
# Ground state
EG0 = 2.*ed+0.5*(U-np.sqrt(U**2.+16.*t**2.))

def test_dedlib_occps():
    espace, egs = build_espace(H,V)
    occp = get_occupation(espace,egs,1e6,0)
    sct = espace[(1,1)]
    expected = get_evecs_occupation(sct.eigvecs[:,[0]],
                                    np.ones((1,)),
                                    sct.states.up,
                                    sct.states.dw,
                                    0)
    np.testing.assert_allclose(occp, expected, 1.)


if __name__ == '__main__':
    import time
    s = time.perf_counter()
    test_dedlib_occps()
    e = time.perf_counter() - s
    print('Elapsed : ', e)