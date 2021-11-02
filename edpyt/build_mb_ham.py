import numpy as np
# Compiled
from numba import njit, prange
from numba.types import float64, uint32, Array

from edpyt.ham_hopping import build_ham_hopping
from edpyt.ham_non_local import build_ham_non_local
from edpyt.ham_local import build_ham_local
from edpyt.shared import unsiged_dt, params


"""
Conventions:
    n : number of levels per spin
    nup : number of up spins
    ndw : number of dw spins
    d : sector size
    dup : sector number of up spin states (all possible permutations of nup in n)
    dwn : sectot number of down spin states (all possible permutations of ndw in n)

"""

def build_mb_ham(H, V, states_up, states_dw, comm=None):
    """Build sparse Hamiltonian of the sector.

    Args:
        H : (np.ndarray, shape=(n,n) or (2,n,n))
            Hamiltonian matrix including the on-site energies
            and the hopping terms on the diagonal and off-diagonal
            entries, respectively. Optional 1st dimension can include
            spin index with mapping {0:up, 1:dw}.
        V : Interaction matrix (n x n).
        nup : number of up spins
        ndw : number of down spins
        comm : if MPI communicator is given the hilbert space
            is assumed to be diveded along spin-down dimension.

    """
    n = H.shape[-1]
    H = np.broadcast_to(H, (2,n,n)) if H.ndim==2 else H
    H.flags.writeable = False

    if isinstance(V, np.ndarray):
        U = V
        Jx = None
        Jp = None
    elif isinstance(V, dict):
        U = V.get('U',None)
        Jx = V.get('Jx', None)
        Jp = V.get('Jp', None)
    # dup = states_up.size
    # dwn = states_dw.size
    # if comm is not None:
    #     size = comm.Get_size()
    #     rank = comm.Get_rank()
    #     dwn_local = dwn//size
    #     d = dwn_local * dup
    # else:    
    #     dwn_local = dwn
    #     d = dup * dwn
    #     rank = 0
    
    operators = list()
    
    operators.append(build_ham_local(H, U, states_up, states_dw, params['hfmode'], params['mu']))
    operators.extend(build_ham_hopping(H, states_up, states_dw))
    if (Jx is not None) and (Jp is not None):
        operators.append(build_ham_non_local(Jx, Jp, states_up, states_dw, operators[0]))

    H.flags.writeable = True

    return operators



@njit(float64(Array(float64, 1, 'C', readonly=False),
      uint32))
def sum_diags_contrib(diags, s):
    # ___                           
    # \         e    n     
    # /__        i    is
    #   s, i
    """Sum diagonal energetic contributions for state s.

    Args:
        diags : onsite energies.
        s : state

    Return:
        res : sum of diagonal contrbutions.

    """
    n = diags.size
    res = 0.
    for i in prange(n):
        if (s>>i)&unsiged_dt(1):
            res += diags[i]
    return res


@njit((Array(float64, 2, 'C', readonly=False),
       Array(float64, 1, 'C', readonly=False),
       uint32[:],uint32[:],float64[:],float64))
def add_onsites(ener_diags, int_diags, states_up, states_dw, vec_diag, hfshift):
    """On-site many-body hamiltonian.

    """
    dup = states_up.size
    dwn = states_dw.size

    # Temporary up contributions for performance.
    vec_diag_up = np.empty(dup)

    for iup in prange(dup):
        sup = states_up[iup]
        vec_diag_up[iup] = sum_diags_contrib(ener_diags[0], sup)

    for idw in prange(dwn):
        sdw = states_dw[idw]
        onsite_energy_dw = sum_diags_contrib(ener_diags[1], sdw)
        for iup in range(dup):
            sup = states_up[iup]
            i = iup + idw*dup
            onsite_int    = sum_diags_contrib(int_diags, sup&sdw)
            onsite_energy = vec_diag_up[iup] + onsite_energy_dw
            vec_diag[i] = onsite_energy + onsite_int + hfshift

