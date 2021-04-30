import numpy as np
from edpyt.symmery import (parity, translation)


def test_translation():
    s = np.uint32(int("00101",base=2))
    N = 5

    np.testing.assert_allclose(
        translation(s, N, 1), 
        np.uint32(int("01010",base=2)))
    
    np.testing.assert_allclose(
        translation(s, N, -1),
        np.uint32(int("10010",base=2)))

    np.testing.assert_allclose(
        translation(s, N, 3),
        np.uint32(int("01001",base=2)))


def test_parity():
    s = np.uint32(int("00101",base=2))
    N = 5

    np.testing.assert_allclose(
        parity(s, N), 
        np.uint32(int("10100",base=2)))