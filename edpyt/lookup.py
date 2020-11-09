import numpy as np
from numba import njit

# Unsigned dtype
from shared import (
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
        return arr_repr

    else:
        raise ValueError(
            "Invalid format type {}.".format(format))


@njit('UniTuple(int64,2)(int64,int64,int64)')
def get_spin_indices(i, dup, dwn):
    """Return spin indices from state i = iup + dup*idw.

    """
    iup = i%dup
    idw = i//dup
    return iup, idw


@njit('int64(int64,int64,int64)')
def get_state_index(iup, idw, dup):
    """Return state index from spin indices i = iup + dup*idw.

    """
    return iup + idw * dup
