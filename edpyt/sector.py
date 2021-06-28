from typing import ValuesView
import numpy as np
from numba import njit

# Binomial coefficient
from scipy.special import binom

# Unsigned dtype
from edpyt.shared import (
    unsiged_dt
)


class OutOfHilbertError(ValueError):
    pass


@njit('void(uint32[:])')
def permutations(states):
    """Generate all permutations of initial bit pattern.

    """
    x = states[0]
    # two = unsiged_dt(2)
    for i in range(1,states.size):
        u = x & -x
        v = u + x
        x = v + (unsiged_dt((v^x)/u)>>2)
        states[i] = x

# @njit('int64(int64,int64)')
def get_sector_dim(n, nup, ndw):
    """Get sector dimension.

    """
    # return binom(n, p)
    return int(binom(n, nup)*binom(n, ndw))

# @njit('uint32[:](int64,int64)')
def generate_states(n, p):
    """Generate states in sector.

    """
    num_states = int(binom(n, p))
    states = np.zeros(num_states,dtype=unsiged_dt)
    initial = unsiged_dt((1<<p)-1)
    states[0] = initial
    permutations(states)
    return states


def get_cdg_sector(n, nup, ndw, ispin):
    """Get N+1 particle sector by adding particle with `ispin` {0:up,1:down}"""
    # Add up electron
    if ispin==0:
        nupJ = nup+1
        ndwJ = ndw
        # Cannot have more spin than spin states
        if nupJ>n:
            raise OutOfHilbertError(f"Out of hilbert.")
    # Add down electron
    elif ispin==1:
        nupJ = nup
        ndwJ = ndw+1
        # Cannot have more spin than spin states
        if ndwJ>n:
            raise OutOfHilbertError(f"Out of hilbert.")
    else:
        raise RuntimeError(f"Invalid spin index {ispin}. Use 0 for up and 1 for down")
    return nupJ, ndwJ


def get_c_sector(nup, ndw, ispin):
    """Get N-1 particle sector by adding particle with `ispin` {0:up,1:down}"""
    # Remove up electron
    if ispin==0:
        nupJ = nup-1
        ndwJ = ndw
        if nupJ<0:
            raise OutOfHilbertError(f"Out of hilbert.")
    # Remove down electron
    elif ispin==1:
        nupJ = nup
        ndwJ = ndw-1
        if ndwJ<0:
            raise OutOfHilbertError(f"Out of hilbert.")
    else:
        raise RuntimeError(f"Invalid spin index {ispin}. Use 0 for up and 1 for down")
    return nupJ, ndwJ