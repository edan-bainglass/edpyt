from edpyt.observs import get_evecs_occupation
import numpy as np
from matplotlib import pyplot as plt
from ase.neighborlist import build_neighbor_list


def draw_skeleton(atoms, axes=None, plane='xy'):
    axes = axes or plt.subplots()[1]
    ax = 'xyz'.index(plane[0]), 'xyz'.index(plane[1])
    pos = atoms.positions[:,ax]
    nl = build_neighbor_list(atoms)
    i = []
    j = []
    for a in atoms:
        nn = nl.get_neighbors(a.index)[0]
        nn = nn[nn>a.index]
        if nn.any():
            i.extend([a.index]*len(nn))
            j.extend(nn.tolist())
    #  xx'yy'
    links = np.vstack([pos[i,0],pos[j,0],pos[i,1],pos[j,1]]).T.reshape(-1,2,2)
    for i in range(len(links)):
        axes.plot(links[i,0],links[i,1],color='black')
    axes.axis('equal')
    axes.axis('off')
    return axes
    
        
def draw_feynman_dyson_orbital(atoms, v, subset=None, axes=None, plane='xy', scale=3):
    if subset is None:
        subset = slice(None)
    positions = atoms[subset].positions
    assert len(positions)==v.size, """
        Invalid orbital. It shoud have the same size
        as the number of atoms.
        """
    axes = draw_skeleton(atoms, axes, plane)
    ax = 'xyz'.index(plane[0]), 'xyz'.index(plane[1])
    v2 = scale*v**2
    for i in range(v.size):
        color = 'r' if v[i]>0 else 'b'
        circle = plt.Circle(positions[i,ax], v2[i], color=color, zorder=10)
        axes.add_artist(circle)
    return axes


def draw_spin_density(atoms, evec, states, n, subset=None, axes=None, plane='xy', scale=3):
    if subset is None:
        subset = slice(None)
    positions = atoms[subset].positions
    assert len(positions)==n, """
        Invalid subset. It shoud have the same size
        as `n`.
        """
    axes = draw_skeleton(atoms, axes, plane)
    nup, ndw = map(lambda nxx: nxx[:,0], 
                   get_evecs_occupation(np.atleast_2d(evec),states.up,states.dw,n))
    ax = 'xyz'.index(plane[0]), 'xyz'.index(plane[1])
    for i in range(n):
        circle = plt.Circle(positions[i,ax], nup[i], color='black', zorder=10)
        axes.add_artist(circle)
    for i in range(n):
        circle = plt.Circle(positions[i,ax], ndw[i], color='gray', zorder=10)
        axes.add_artist(circle)
    return axes