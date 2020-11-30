import numpy as np
from numba import njit

# Unsigned dtype
from edpyt.shared import (
    unsiged_dt
)


@njit()
def count_bits(s, n):
    """Count the number of '1' bits in a state s.

    """
    bits = 0
    for i in range(n):
        bits += (s>>i)&unsiged_dt(1)
    return bits


def binrep(i, n, format="array"):
    """Return binary representation in vector format.

    Args:
        format : "string" ("001000..") or "array" ([0,0,1,0,...])
    """

    str_repr = np.binary_repr(i, n)
    if format in ["string",'str']:
        return str_repr

    arr_repr = np.fromiter(str_repr,dtype='S1').astype(unsiged_dt)
    if format in ["array","arr"]:
        return arr_repr[::-1]

    else:
        raise ValueError(
            "Invalid format type {}.".format(format))


@njit(['UniTuple(int64,2)(int64,int64,int64)',
       'UniTuple(int32,2)(int32,int32,int32)'])
def get_spin_indices(i, dup, dwn):
    """Implements map i = iup + dup*idw -> (iup,idw).

    """
    iup = i%dup
    idw = i//dup
    return iup, idw


@njit(['int64(int64,int64,int64)',
       'int32(int32,int32,int32)'])
def get_state_index(iup, idw, dup):
    """Implements map (iup,idw) -> i = iup + dup*idw.

    """
    return iup + idw * dup


@njit(['int64(uint32[:],uint32)',
       'int32(uint32[:],uint32)'])
def binsearch(a, s):
    """Binary search the element s in array a. Return index of s."""
    lo = -1
    hi = a.size
    while hi-lo > 1:
        m = (lo+hi)>>1
        if a[m] <= s:
            lo = m
        else:
            hi = m

    if lo == -1 or a[lo] != s:
        return -1
    else:
        return lo


# @njit('UniTuple(int64,2)(int64)')
def get_num_spins(isct, n):
    """Implements map isct -> (nup,ndw)

    """
    return np.unravel_index(isct,(n+1,n+1))


# @njit('int64(int64,int64,int64)')
def get_sector_index(nup, ndw, n):
    """Implements map (nup,ndw) -> isct

    """
    return np.ravel_multi_index((nup,ndw),(n+1,n+1))
