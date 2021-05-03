import numpy as np
from functools import singledispatch
from collections import namedtuple

from edpyt.shared import unsiged_dt
uOne = unsiged_dt(1)


def parity(s, n):
    """Parity is the reflection of a state w.r.t. the middle of the chain."""
    out = 0
    f = n-1
    #
    out ^= (s&uOne)
    s >>= 1
    while(s):
            out <<= 1
            out ^= (s&uOne)
            s >>= 1
            f -= 1
    #
    out <<= f
    return out


def translation(s, n, shift):
    """Translate state by shift sites """
    period = n # periodicity/cyclicity of translation
    smax = (uOne<<n)-uOne # largest integer allowed to appear in the basis
    #
    l = (shift+period)%period
    s1 = (s >> unsiged_dt(period - l))
    s2 = ((s << l) & smax)
    #
    return (s2 | s1)


Symmetry = namedtuple('Symmetry',['chi','apply','n','args'])


def check_state(s, symmetries):
    """Get representative for state s and index of symmetries that 
    maps s to the representative, i.e. symmetries[i].apply(s)->r.
    """
    r = s
    degeneracy = 0
    sum_chi = 0.
    for i, symm in enumerate(symmetries):
        ss = symm.apply(s)
        if ss<r: # representative already in the list.
            return -1
        if ss==s:
            degeneracy += 1
            sum_chi += symm.chi
    if abs(sum_chi)<1e-7: # representative not allowed, norm=0.
        return -1
    return degeneracy


def add_state(s, reprs, degens, symmetries):
    x = check_state(s, symmetries)
    if x>0:
        reprs.append(s)
        degens.append(x)


def get_representative(s, symmetries):
    r = s
    indx = 0
    for i, symm in enumerate(symmetries):
        ss = symm.apply(s)
        if ss<r: # representative already in the list.
            r = ss
            indx = i
    return r, i


def find_state(s, a):
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


def apply_symm(diag_ops, offdiag_ops, reprs, degens):
    pass
    # for r, degeneracy in zip(reprs, degens):
