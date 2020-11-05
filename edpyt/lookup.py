import numpy as np
from numba import njit

# Unsigned dtype
from shared import (
    unsiged_dt
)


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


def get_spin_indices(i, dup, dwn):
    """Return spin indices in state i = iup + dup*idw.

    """
    idw, iup = np.unravel_index(i, (dwn, dup))
    return iup, idw
