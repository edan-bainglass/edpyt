import numpy as np
from numba import njit, types

# Unsigned dtype
from shared import (
    unsiged_dt
)

@njit
def flip(s, pos):
    """Flip spin at position pos.

    """
    s_out = (unsiged_dt(1)<<pos)^s
    return s_out


@njit
def fsgn(s, pos):
    """Fermionic sign of state s.

    """
    bits = 0
    for k in range(pos):
        bits += (s>>k)&1
    return 1-2*(bits%2)


def cdg(s, pos):
    """Fermionic creation operator.

    Args:
        s : initial state
        pos : flip at position

    Return:
        s_out : c_pos |s> = c_pos |0,...,1_pos,> = |0,...,0_pos,>
        fermionic sign : (-1)^v (v = sum_{i<pos} 1_i)
    """
    s_out = flip(s, pos)
    fermionic_sgn = fsgn(s, pos)

    assert s_out > s, "c^+ error : c|0,...1_pos,>"

    return fermionic_sgn, s_out


def c(s, pos):
    """Fermionic annihilation operator.

    Args:
        s : initial state
        pos : flip at position

    Return:
        s_out : c^+_pos |s> = c^+_pos |0,...,0_pos,..> = |0,...,1_pos,..>
        fermionic sign : (-1)^v (v occupied states to the left of pos)
    """
    s_out = flip(s, pos)
    fermionic_sgn = fsgn(s, pos)

    assert s_out < s, "c error : c|0,...0_pos,>"

    return fermionic_sgn, s_out
