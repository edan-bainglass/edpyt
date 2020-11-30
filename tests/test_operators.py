from edpyt.operators import (
    fsgn,
    flip,
    c,
    cdg,
    check_full,
    check_empty
)

import numpy as np

def test_fsgn():

    s = int("101101",base=2)
    assert fsgn(s, np.int32(2)) == (-1)**1
    assert fsgn(s, 3) == (-1)**2
    assert fsgn(s, 6) == (-1)**4


def test_flip():

    s = int("101101",base=2)
    assert flip(s, 2) == int("101001",base=2)
    assert flip(s, 5) == int("001101",base=2)


def test_fer_algebra():

    s = int("101101",base=2)
    i = 0
    j = 1
    # {c_i,c^+_j} = c_i c^+_j + c^+_j c_i = \delta_ij
    #
    tsgn, t = c(s, i)
    fsgn, f1 = cdg(t, j)
    sgn1 = tsgn * fsgn
    #
    tsgn, t = cdg(s, j)
    fsgn, f2 = c(t, i)
    sgn2 = tsgn * fsgn
    #
    assert sgn1*f1 + sgn2*f2 == 0


def test_check_full():

    s = int("101101",base=2)
    assert check_full(s, 0)
    assert check_full(s, 2)

def test_check_empty():

    s = int("101101",base=2)
    assert check_empty(s, 1)
    assert check_empty(s, 4)
    assert not check_empty(s, 3)
