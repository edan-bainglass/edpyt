from collections import defaultdict
import numpy as np

from edpyt.gf_exact import project_exact_up, project_exact_dw, Gf
from edpyt.sector import OutOfHilbertError, get_cdg_sector

from edpyt.operators import c, cdg

class Gf2:
    
    def __init__(self, gf):
        self.shape = gf.shape
        self.gf = gf
        
    
    def __call__(self, e, eta):
        shape = self.shape
        G = np.empty(shape + (e.size,),complex)
        for p1, p2 in np.ndindex(shape):
            G[p1,p2] = self.gf[p1,p2](e, eta)
        return G

def build_gf2_exact(H, V, espace, beta, egs=0., ispin=0):
    n = H.shape[-1]
    project = [project_exact_up, project_exact_dw][ispin]

    lambdas = defaultdict(list)
    qs = defaultdict(list)
    gf = np.ndarray(shape=(n,n),dtype=object)
    
    for (nupI,ndwI), sctI in espace.items():
        try: # Add spin (N+1 sector)
            nupJ, ndwJ = get_cdg_sector(n, nupI, ndwI, ispin)
        except OutOfHilbertError: # More spin than spin states
            continue
        sctJ = espace[(nupJ,ndwJ)]
        EI = (sctI.eigvals-egs)[None,:]
        EJ = (sctJ.eigvals-egs)[:,None]
        exponents = np.exp(-beta*EJ) + np.exp(-beta*EI)
        for p1, p2 in np.ndindex(n,n):
            b2J = project(p1, n, c, sctJ, sctI).T * project(p2, n, cdg, sctI, sctJ)
            lambdas[p1,p2].extend((EJ - EI).flatten())
            qs[p1,p2].extend((b2J * exponents).flatten())
            
    Z = sum(np.exp(-beta*(sct.eigvals-egs)).sum() for
        (nup, ndw), sct in espace.items())
    
    for p1, p2 in np.ndindex(n,n):
        gf[p1,p2] = Gf(qs[p1,p2], lambdas[p1,p2], Z)
    
    return Gf2(gf)