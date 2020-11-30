import numpy as np
from numba import njit
from scipy.optimize import fsolve
from scipy.misc import derivative

# Reference:

#  Kondo physics of the Anderson impurity model by Distributional ExactDiagonalization.
#  S. Motahari,  R.  Requist,  and  D.  Jacob


def sample(p, rng, x0, n):
    """Sample n random poles using probability p.

    Args:
        p : (callable) probability distribution.
        rng : range of sampling around x0.
        n : # of poles.

    Returns:
        poles : sampled poles.
    """
    z = rng * np.random.random(int(1e6)) - rng/2 - x0
    probs = p(z)
    probs /= probs.sum()
    poles = np.random.choice(z, n, p=probs)
    poles.sort()
    return poles


def build_gf0(poles):
    """Build non-interacting Green's function.

    """
    #           __ n
    #  0       \                1/n
    # G (z)                  ---------
    #          /__ i = 0       z - poles
    #                                   i
    n = poles.size
    def gf0(z):
        z = np.array(z, ndmin=1)
        return 1/n * np.reciprocal(z[:,None]-poles[None,:]).sum(1)
    gf0.poles = poles
    gf0.n = poles.size
    gf0.derivative = lambda z: np.sum(-1/(z-gf0.poles)**2)/gf0.n
    return gf0


def build_gfimp(gf0):
    """Fit SAIM model.

    Returns:
        G(z) : (callable) impurity Green's function.
        - G.ek = onsite bath energies.
        - G.vk2 = imp-bath square hoppings.
        - G.e0 = impurity energy level.
    """
    #
    #  0      !     1
    # G (z)   =  ----------  __ n-1           2
    #            z - e0 -    \               Vk
    #                                     -------
    #                        /__k = 0      z - ek
    poles = gf0.poles
    n = gf0.n
    ek = np.zeros(n-1)
    vk2 = np.zeros(n-1)
    for i in range(n-1):
        ek[i] = fsolve(gf0, (poles[i+1]+poles[i])/2)
        vk2[i] = -1/gf0.derivative(ek[i])#(gf0, ek[i], dx=1e-6)
    e0 = poles.mean()
    def gfimp(z):
        z = np.array(z, ndmin=1)
        delta = lambda z: vk2[None,:] * np.reciprocal(z[:,None]-ek[None,:])
        return np.reciprocal(z-e0-delta(z).sum(1))
    gfimp.e0 = e0
    gfimp.ek = ek
    gfimp.vk2 = vk2
    return gfimp


@njit()
def get_occupation(vector, states_up, states_dw, n):
    """Count particles in eigen-state vector (size=dup x dwn).

    """
    N = 0.
    occps = vector**2
    d = states_up.size*states_dw.size
    dup = states_up.size
    dwn = states_dw.size

    for i in range(d):
        iup = i%dup
        idw = i//dup
        sup = states_up[iup]
        sdw = states_dw[idw]
        for j in range(0,n):
            if (sup>>j)&np.uint32(1):
                N += occps[i]
            if (sdw>>j)&np.uint32(1):
                N += occps[i]
    return N


def keep_gs(espace, egs):
    """Keep sectors containing ground state.

    """
    delete = []
    for (nup, ndw), sct in espace.items():
        diff = np.abs(sct.eigvals[0]-egs) < 1e-7
        if ~diff.any():
            delete.append((nup,ndw))
    for k in delete:
        espace.pop(k)


def build_siam(H, V, U, gfimp):
    vk = np.sqrt(gfimp.vk2)
    n = H.shape[0]
    H[1:,0] = H[0,1:] = - vk
    H.flat[(n+1)::(n+1)] = gfimp.ek
    H[0,0] = gfimp.e0
    V[0,0] = U
