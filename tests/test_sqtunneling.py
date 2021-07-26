import numpy as np
from collections import defaultdict
from edpyt.espace import Sector, build_empty_sector
from edpyt.sqtunneling import (
    build_transition_elements, 
    build_rate_matrix,
    build_transition_matrix, project_sector)
from edpyt.rates import stationary_solution


# Ref.
# http://www.capri-school.eu/capri05/lectures/flensberg12.pdf
# https://www.cambridge.org/core/services/aop-cambridge-core/content/view/955EAEB199219F9263F8AE61979775C6/9781139164313c3_p51-80_CBO.pdf/selfconsistent_field.pdf

n = 1
eps = 0.2
U = 0.25
beta = 1000.
mu = [0.,0.]
A = np.ones((2,n))


def build_espace(eps, U):
    """Solve anderson model."""
    n = 1
    espace = defaultdict(Sector)
    eigvals = [0.,eps,eps,eps+eps+U]
    for i, (nup, ndw) in enumerate(np.ndindex((n+1, n+1))):
        sct = build_empty_sector(n, nup, ndw)
        sct.eigvecs = np.array(1., ndmin=2)
        sct.eigvals = np.array(eigvals[i], ndmin=1)
        espace[(nup,ndw)] = sct
    egs = np.min(eigvals)
    return espace, egs


def test_project_sector():
    """Purpose is to check if the arrival sectors are correct."""    
    espace, egs = build_espace(eps, U)
    
    gfdict = {}
    project_sector(n, 0, 0, espace, gfdict)
    expected = [((1, 0, 0), (0, 0, 0)), ((0, 1, 0), (0, 0, 0))]
    assert all([iD in expected for iD in gfdict.keys()])

    gfdict = {}
    project_sector(n, 1, 0, espace, gfdict)
    expected = [((0, 0, 0), (1, 0, 0)), ((1, 1, 0), (1, 0, 0))]
    assert all([iD in expected for iD in gfdict.keys()])

    gfdict = {}
    project_sector(n, 0, 1, espace, gfdict)
    expected = [((0, 0, 0), (0, 1, 0)), ((1, 1, 0), (0, 1, 0))]
    assert all([iD in expected for iD in gfdict.keys()])

    gfdict = {}
    project_sector(n, 1, 1, espace, gfdict)
    expected = [((0, 1, 0), (1, 1, 0)), ((1, 0, 0), (1, 1, 0))]
    assert all([iD in expected for iD in gfdict.keys()])

    
def test_rate_matrx():
    """Purpose is to test if steady-state probability corresponds 
    to ground state."""
    
    def solve(Vg):
        espace, egs = build_espace(eps+Vg, U)
        gfdict = {}
        for nup,ndw in [(0,0),(1,0),(0,1),(1,1)]:
            project_sector(n, nup, ndw, espace, gfdict)
        W = build_rate_matrix(gfdict, beta, mu, A)
        return stationary_solution(W)
    
    # Empty
    np.testing.assert_allclose(solve(Vg=0.), [1.,0.,0.,0.])
    # Degenerate empty and singly-occupied
    np.testing.assert_allclose(solve(Vg=-eps), [0.333,0.333,0.333,0.], atol=1e-3)
    # Degenerate singly- and doubly-occupied
    np.testing.assert_allclose(solve(Vg=-(eps+U)), [0.,0.333,0.333,0.333], atol=1e-3)
    

def test_transition_elems():
    """Same as test_rate_matrix but use only ground state 
    as starting sector."""
    
    def solve(Vg):
        espace, egs = build_espace(eps+Vg, U)
        gfdict = build_transition_elements(n, egs, espace)
        W = build_rate_matrix(gfdict, beta, mu, A)
        return stationary_solution(W)
    
    # Empty
    np.testing.assert_allclose(solve(Vg=0.), [1.,0.,0.])
    # Degenerate empty and singly-occupied
    np.testing.assert_allclose(solve(Vg=-eps), [0.333,0.333,0.333,0.], atol=1e-3)
    # Degenerate singly- and doubly-occupied
    np.testing.assert_allclose(solve(Vg=-(eps+U)), [0.,0.333,0.333,0.333], atol=1e-3)
    

def test_transition_matrix():
    """Same as test_rate_matrix but use only ground state 
    as starting sector."""
    
    espace, egs = build_espace(eps, U)
    gfdict = {}
    for nup,ndw in [(0,0),(1,0),(0,1),(1,1)]:
        project_sector(n, nup, ndw, espace, gfdict)
    W = build_rate_matrix(gfdict, beta, mu, A)
    P = stationary_solution(W)
    T = build_transition_matrix(gfdict, beta, mu[0], A[0])
    I = T.dot(P).sum()
    
    map = {0:(0,0,0),1:(0,1,0),2:(1,1,0)}
    args = A[0],beta,mu[0]
    np.testing.assert_allclose(
        I, 
        2*P[0]*gfdict[map[1],map[0]](*args)
        - 2*P[2]*gfdict[map[1],map[2]](*args)
        + P[1]*(-gfdict[map[0],map[1]](*args)+gfdict[map[2],map[1]](*args))
        )
    

if __name__=='__main__':
    test_project_sector()
    test_rate_matrx()
    test_transition_elems()
    test_transition_matrix()    