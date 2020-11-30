import numpy as np
from numba import njit

# Binomial coefficient
from scipy.special import binom

# Unsigned dtype
from edpyt.shared import (
    unsiged_dt
)


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
def get_sector_dim(n, p):
    """Get sector dimension.

    """
    # return binom(n, p)
    return int(binom(n, p))

# @njit('uint32[:](int64,int64)')
def generate_states(n, p):
    """Generate states in sector.

    """
    num_states = get_sector_dim(n, p)
    states = np.zeros(num_states,dtype=unsiged_dt)
    initial = unsiged_dt((1<<p)-1)
    states[0] = initial
    permutations(states)
    return states
