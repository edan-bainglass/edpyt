import numpy as np
from numba import njit, types

# Unsigned dtype
from edpyt.shared import (
    unsiged_dt
)

@njit(['uint32(uint32,int64)',
       'uint32(uint32,int32)'])
def flip(s, pos):
    """Flip spin at position pos.

    """
    s_out = (unsiged_dt(1)<<pos)^s
    return s_out


@njit(['int64(uint32,int64)',
       'int32(uint32,int32)'])
def fsgn(s, pos):
    """Fermionic sign of state s.

    """
    bits = 0
    for k in range(pos):
        bits += (s>>k)&unsiged_dt(1)
    return 1-2*(bits%2)


@njit(['Tuple((int64,uint32))(uint32,int64)',
       'Tuple((int32,uint32))(uint32,int32)'])
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

    assert s_out > s, "c^+ error : c^+|0,...1_pos,>"

    return fermionic_sgn, s_out


@njit(['Tuple((int64,uint32))(uint32,int64)',
       'Tuple((int32,uint32))(uint32,int32)'])
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


@njit(['int32(uint32,int64)',
       'int32(uint32,int32)'])
def check_full(s, pos):
    """Particle count operator.

    """
    return (s>>pos)&unsiged_dt(1)


@njit(['int32(uint32,int64)',
       'int32(uint32,int32)'])
def check_empty(s, pos):
    """Hole count operator.

    """
    return not check_full(s, pos) #e(s>>pos)^unsiged_dt(1)
