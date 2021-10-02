import numpy as np
from numba import njit, float64

# Unsigned dtype
from edpyt.shared import unsigned_one


@njit(['uint32(uint32,int64)',
       'uint32(uint32,int32)'])
def flip(s, pos):
    """Flip spin at position pos.

    """
    s_out = (unsigned_one<<pos)^s
    return s_out


@njit(['int64(uint32,int64, int64)',
       'int32(uint32,int32, int64)'])
def fsgn(s, pos, n):
    """Fermionic sign of state s.

    """
    bits = 0
    for k in range(pos+1,n):
        bits += (s>>k)&unsigned_one
    return 1-2*(bits%2)


@njit(['int64(uint32,int64,int64)',
       'int32(uint32,int32,int32)'])
def get_parity(s, i, j):
    """Count the '1' bits in a state s between to sites i and j.    """
    # Put i, j in order.
    i, j = sorted((i, j))

    bits = 0
    for k in range(i+1, j):
        bits += (s>>k)&1
    return 1-2*(bits%2)


@njit(['Tuple((int64,uint32))(uint32,int64,int64)',
       'Tuple((int32,uint32))(uint32,int32,int64)'])
def cdg(s, pos, n):
    """Fermionic creation operator.

    Args:
        s : initial state
        pos : flip at position

    Return:
        s_out : c_pos |s> = c_pos |0,...,1_pos,> = |0,...,0_pos,>
        fermionic sign : (-1)^v (v = sum_{i<pos} 1_i)
    """
    s_out = flip(s, pos)
    fermionic_sgn = fsgn(s, pos, n)

    assert s_out > s, "c^+ error : c^+|0,...1_pos,>"

    return fermionic_sgn, s_out


@njit(['Tuple((int64,uint32))(uint32,int64,int64)',
       'Tuple((int32,uint32))(uint32,int32,int64)'])
def c(s, pos, n):
    """Fermionic annihilation operator.

    Args:
        s : initial state
        pos : flip at position

    Return:
        s_out : c^+_pos |s> = c^+_pos |0,...,0_pos,..> = |0,...,1_pos,..>
        fermionic sign : (-1)^v (v occupied states to the left of pos)
    """
    s_out = flip(s, pos)
    fermionic_sgn = fsgn(s, pos, n)

    assert s_out < s, "c error : c|0,...0_pos,>"

    return fermionic_sgn, s_out


@njit(['Tuple((int64,uint32))(uint32,int64,int64)',
       'Tuple((int32,uint32))(uint32,int32,int32)'])
def cdgc(s, i, j):
    f = (s^(unsigned_one<<j))^(unsigned_one<<i)
    sgn = get_parity(s, i, j)
    return sgn, f


@njit(['uint32(uint32,int64)',
       'uint32(uint32,int32)'])
def check_full(s, pos):
    """Particle count operator.

    """
    return (s>>pos)&unsigned_one


@njit(['uint32(uint32,int64)',
       'uint32(uint32,int32)'])
def check_empty(s, pos):
    """Hole count operator.

    """
    return not check_full(s, pos) #e(s>>pos)^unsigned_one


n_op = check_full
ndg_op = check_empty