#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fit
from edpyt.espace import build_espace
from edpyt.gf_lanczos import build_gf_lanczos
from edpyt.lookup import get_sector_index, get_spin_indices
import progressbar
from matplotlib import pyplot as plt
import numpy as np

# In[2]:


def lorentzian_function(gamma, z0):
    def inner(z):
        return 1/np.pi * (0.5*gamma) / ((z-z0)**2 + (0.5*gamma)**2)
    inner.z0 = z0
    inner.gamma = gamma
    return inner


# In[3]:


lorentz = lorentzian_function(2*0.3, 0.)


# In[4]:


energies = np.linspace(-2,2,1001,endpoint=True)
#plt.plot(energies, lorentz(energies))


# In[5]:


n = 4
N = 10
M = n * N

def sample(p, rng, x0, n):
    z = rng * np.random.random(int(1e6)) - rng/2 - x0
    probs = p(z)
    probs /= probs.sum()
    poles = np.random.choice(z, n, p=probs)
    poles.sort()
    return poles


# In[6]:


poles = sample(lorentz, 10., 0., n)


# In[7]:


#plt.plot(poles, lorentz(poles), 'o')


# In[8]:


def build_gf0(poles):
    def gf0(z):
        z = np.array(z, ndmin=1)
        return 1/n * np.reciprocal(z[:,None]-poles[None,:]).sum(1)
    gf0.poles = poles
    gf0.n = poles.size
    return gf0


# In[9]:


gf0 = build_gf0(poles)


# In[10]:


eta = 0.02
#plt.plot(energies, gf0(energies+1.j*eta).real)
#plt.plot(poles, np.zeros(poles.size), 'o')


# In[11]:


from scipy.optimize import fsolve
from scipy.misc import derivative

def build_gfimp(gf0):
    poles = gf0.poles
    n = gf0.n
    ek = np.zeros(n-1)
    vk2 = np.zeros(n-1)
    for i in range(n-1):
        ek[i] = fsolve(gf0, (poles[i+1]+poles[i])/2)
        vk2[i] = -1/derivative(gf0, ek[i], dx=1e-6)
    e0 = poles.mean()
    def gfimp(z):
        z = np.array(z, ndmin=1)
        delta = lambda z: vk2[None,:] * np.reciprocal(z[:,None]-ek[None,:])
        return np.reciprocal(z-e0-delta(z).sum(1))
    gfimp.e0 = e0
    gfimp.ek = ek
    gfimp.vk2 = vk2
    return gfimp


# e0 = (gfimp.vk2/gfimp.ek).sum() + n * ((1/gf0.poles).sum())**-1

# In[12]:


gfimp = build_gfimp(gf0)


# In[13]:


eta = 0.02
#plt.plot(energies, gf0(energies+1.j*eta).real)
#plt.plot(energies, gfimp(energies+1.j*eta).real)


# In[14]:


eta = 0.02
#plt.plot(energies, gf0(energies+1.j*eta).imag)
#plt.plot(energies, gfimp(energies+1.j*eta).imag)


# In[15]:


H = np.zeros((n,n))
V = np.zeros((n,n))

def build_H(H, V, e0, U, ek, vk):
    n = H.shape[0]
    H[1:,0] = H[0,1:] = vk
    H.flat[(n+1)::(n+1)] = ek
    H[0,0] = e0
    V[0,0] = U
    
#vk = np.square(gfimp.vk2)
#build_H(H, V, gfimp.e0, 0., gfimp.ek, -vk)


# In[16]:


neig = np.ones((n+1)*(n+1)) * 1
espace, egs = build_espace(H, V, neig)


# In[17]:


def keep_gs(espace, egs):
    delete = []
    for (nup, ndw), sct in espace.items():
        diff = np.abs(sct.eigvals[0]-egs) < 1e-7
        if ~diff.any():
            delete.append((nup,ndw))
    for k in delete:
        espace.pop(k)


# In[18]:


keep_gs(espace, egs)


# In[19]:


#gf0imp = build_gf_lanczos(H, V, espace, 0.)


# In[20]:


from numba import njit

@njit()
def get_occupation(vector, states_up, states_dw):
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


# In[21]:


n = 4
N = int(5e4)
M = n * N
H = np.zeros((n,n))
V = np.zeros((n,n))
U = 3.
eta = 0.02
wr = np.load('mesh_pm5.npy') #energies
wi = eta*np.abs(wr)
w = wr + 1.j*wi


# In[22]:


sigma = np.zeros_like(w)


# In[23]:


from edpyt.shared import params
params['hfmode'] = False #True


# In[24]:


count = 0
for _ in progressbar.progressbar(range(N)):
    found = False
    while not found:
        poles = sample(lorentz, 10., 0., n)
        gf0 = build_gf0(poles)
        gfimp = build_gfimp(gf0)
        vk = np.sqrt(gfimp.vk2)
        build_H(H, V, gfimp.e0, 0., gfimp.ek, -vk)
        neig = np.ones((n+1)*(n+1)) * 1
        espace, egs = build_espace(H, V, neig)
        keep_gs(espace, egs)
        sct = next(v for v in espace.values())
        if sct.eigvecs.ndim < 2: continue
        evec = sct.eigvecs[:,0]
        N0 = get_occupation(evec,sct.states.up,sct.states.dw)
        V[0,0] = U
        H[0,0] -= U/2
        espace, egs = build_espace(H, V, neig)
        keep_gs(espace, egs)
        sct = next(v for v in espace.values())
        evec = sct.eigvecs[:,0]
        Nv = get_occupation(evec,sct.states.up,sct.states.dw)
        if np.allclose(Nv,N0):
            try:
                gf = build_gf_lanczos(H, V, espace, 0.)
            except:
                continue
            sigma += np.reciprocal(gf0(w))-np.reciprocal(gf(wr,wi))
            found = True
            count += 1



# In[ ]:


np.save('sigma', sigma/count)



# In[28]:


gf = 1 / (w - sigma/count + 0.3j)
plt.plot(wr, -1/np.pi * gf.imag)
plt.savefig('ded.png', dpi=300)
plt.close()

# In[ ]:




