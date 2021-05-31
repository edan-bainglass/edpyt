import numpy as np
from edpyt.operators import n_op


def get_occupation(espace, egs, beta, n):
    """Count particles in espace eigen-state vectors (size=dup x dwn).

    """
    nup = np.zeros(n)
    ndw = np.zeros(n)
    Z = 0. # Partition function
    for sct in espace.values():
        exps = np.exp(-beta*(sct.eigvals-egs))
        evecs = sct.eigvecs
        nup_nev, ndw_nev = get_evecs_occupation(evecs,sct.states.up,sct.states.dw,n,exps)
        nup += nup_nev.sum(1)
        ndw += ndw_nev.sum(1)
        Z += exps.sum()
    nup /= Z
    ndw /= Z
    return nup, ndw


def get_evecs_occupation(evecs, states_up, states_dw, n, coeffs=None, comm=None):
    """Count spin occupations in eigen-state vectors.

    Args:
        evecs : (np.ndarray, order='F' or 'C')
            eigen-state vectors in hilbert space basis (sector).
        states_up(dw) : hilbert space basis (sector).
        n : number of sites.
        coeffs : (optional) weight each eigen-state.
        comm : if MPI communicator is given the hilbert space
            is assumed to be diveded along spin-down dimension.

    Returns:
        nup, ndw : (np.ndarray, shape=(# sites, # eigen-states))
            up and down occupations per site-state.
    """
    #                             ___ d (# sector dim)   
    #                             \        
    #  N(j)                 =             < v(j)    | n         | v(j)   >  c(j)
    #   l,spin (site index)       /__ i         il     l,spin        il    
    if comm is None:
        return _get_evecs_occupation(evecs, states_up, states_dw, n, coeffs)
    else:
        return _get_evecs_occupation_mpi(evecs, states_up, states_dw, n, comm, coeffs)


def _get_evecs_occupation(evecs, states_up, states_dw, n, coeffs=None):
    """
    NOTE: this approach is valid because of the property 
               !
        (a-b)*c = a*c-c*b

    Example:

        states            site-0         site-1

    ((0, 1), (0, 1))   (0-0)*c[0,0]   (1-1)*c[0,1]
    ((0, 1), (1, 0))   (0-1)*c[1,0]   (1-0)*c[1,1]
    ((1, 0), (0, 1))   (1-0)*c[2,0]   (0-1)*c[2,1]
    ((1, 0), (1, 0))   (1-1)*c[3,0]   (0-0)*c[3,1]

    c.shape = (2,2)
    
    sz(site-0) = sum(column # 0) = 
        
        ( 0*c[:,0] + 1*c[:,1] ) - ( 0*c[0,:] + 1*c[1,:] )
        
                  up                        dw
    """
    dup = states_up.size
    dwn = states_dw.size
    nev = evecs.shape[1]
    nup = np.zeros((n,nev))
    ndw = np.zeros((n,nev))
    occps = np.zeros(nev) # internal
    
    if evecs.flags.f_contiguous:
        evecs = evecs.T
        evecs.shape = (nev, dwn, dup)
        if coeffs is None:
            contract_up = lambda iup: np.einsum('ij,ji->i',evecs[:,:,iup],evecs[:,:,iup].T,out=occps,optimize=True)
            contract_dw = lambda idw: np.einsum('ij,ji->i',evecs[:,idw,:],evecs[:,idw,:].T,out=occps,optimize=True)
        else:
            contract_up = lambda iup: np.einsum('i,ij,ji->i',coeffs,evecs[:,:,iup],evecs[:,:,iup].T,out=occps,optimize=True)
            contract_dw = lambda idw: np.einsum('i,ij,ji->i',coeffs,evecs[:,idw,:],evecs[:,idw,:].T,out=occps,optimize=True)
    else:
        evecs.shape = (dwn, dup, nev)
        if coeffs is None:
            contract_up = lambda iup: np.einsum('ji,ij->i',evecs[:,iup,:],evecs[:,iup,:].T,out=occps,optimize=True)
            contract_dw = lambda idw: np.einsum('ji,ij->i',evecs[idw,:,:],evecs[idw,:,:].T,out=occps,optimize=True)
        else:
            contract_up = lambda iup: np.einsum('i,ji,ij->i',coeffs,evecs[:,iup,:],evecs[:,iup,:].T,out=occps,optimize=True)
            contract_dw = lambda idw: np.einsum('i,ji,ij->i',coeffs,evecs[idw,:,:],evecs[idw,:,:].T,out=occps,optimize=True)
    
    for iup in range(dup):
        sup = states_up[iup]
        # contract up component
        occps = contract_up(iup)
        for k in range(n):
            nup[k,:] += n_op(sup, k)*occps

    for idw in range(dwn):
        sdw = states_dw[idw]
        # contract down component
        occps = contract_dw(idw)
        for k in range(n):
            ndw[k,:] += n_op(sdw, k)*occps

    return nup, ndw


def _get_evecs_occupation_mpi(evecs, states_up, states_dw, n, comm, coeffs=None):
    """MPI support for spin particle count.
    
    NOTE: states_dw is the full hilbert down space, NOT the local.
    """
    from mpi4py import MPI
    dwn_local = states_dw.size//comm.size
    nup_local, ndw_local = _get_evecs_occupation(evecs, states_up,
        states_dw[comm.rank*dwn_local:(comm.rank+1)*dwn_local], n, coeffs)
    nup = np.empty_like(nup_local)
    ndw = np.empty_like(ndw_local)
    comm.Allreduce([nup_local, nup_local.size], [nup, nup.size], op=MPI.SUM)
    comm.Allreduce([ndw_local, ndw_local.size], [ndw, ndw.size], op=MPI.SUM)
    return nup, ndw