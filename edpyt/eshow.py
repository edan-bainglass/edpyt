from collections import defaultdict
from edpyt.lookup import binrep

def pdisplay(espace, ax=None):
    """Display charge spectrum.
    
    Group Sz sectors into charge sectors and 
    diplay corresponding spectrum.
    """
    from matplotlib import pyplot as plt
    ax = ax or plt
    pspace = defaultdict(list)
    for key, sct in espace.items():
        pspace[sum(key)].extend(sct.eigvals.tolist())
    xavg = sum(map(lambda x: [x[0]]*len(x[1]), pspace.items()), [])
    y = sum(pspace.values(), [])
    ax.hlines(y, [x-0.2 for x in xavg], [x+0.2 for x in xavg])

def eprint(espace, n):
    """Print Hilbert space."""
    def fockstates(sct):
        
        states = []
        for sdw in sct.states.dw:
            for sup in sct.states.up:
                states.append( fock(sup,sdw,n) )

        return states

    estates = []
    for ns, sct in espace.items():
        sates = fockstates(sct)
        evecs = [' + '.join([v+s for v, s in zip(evec.round(3).astype(str), sates)])
                for evec in sct.eigvecs.T]
        estates.extend([(ns,e,v) for e,v in zip(sct.eigvals, evecs)])
    estates = sorted(estates, key=lambda estate: estate[1]) # sort by enrgy
    for ns, e, v in estates:
        print(f'{ns}: {e:.3E}', v, sep='\n', end='\n\n')
    # print(*estates,sep='\n')        
        # print(ns,sct.eigvals.reshape(-1,1),evecs,sep='\n',end='\n\n')        

def fock(sup, sdw, n):
    """Represent fock state.
    
    Args: 
        sup : spin-up state.
        sdw : spin-dw state.
        n : number of sites.
        
    Example:
    In [1]: from edpyt.eshow import fock
    In [2]: from edpyt.lookup import binrep

    In [3]: fock(3,1,2)
    Out[3]: '|↑↓,↑>'

    In [4]: binrep(3,2)
    Out[4]: array([1, 1], dtype=uint32)

    In [5]: binrep(1,2)
    Out[5]: array([1, 0], dtype=uint32)
    """
    uparrow = u'\u2191'
    downarrow = u'\u2193'
    
    nup = binrep(sup, n)
    ndw = binrep(sdw, n)
    
    state = '|'
    for i in range(n):
        a, b = nup[i], ndw[i]
        if not (a or b):
            state += 'o'
        if a:
            state += uparrow
        if b:
            state += downarrow
        state += ','
    return state[:-1]+'>'