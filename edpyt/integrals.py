import numpy as np
from mpmath import psi
from scipy.constants import hbar as _hbar, eV

hbar = _hbar/eV # Units of electron volt.

# https://www.weizmann.ac.il/condmat/oreg/sites/condmat.oreg/files/uploads/Thesises/carmiphd.pdf

def _Psi(n, a, b, beta):
    """Polygamma function of order n."""
    return psi(n, 0.5 + (1.j*beta)/(2.*np.pi)*(a-b))

def _nB(eps, beta):
    """Bose distribution."""
    return 1./(np.exp(beta*eps)-1)

def _I1(C, eps, mu, beta):
    f = C**2 * beta/(2*np.pi) * np.imag(
        _Psi(1,mu[1],eps,beta) 
      - _Psi(1,mu[0],eps,beta)
    )
    with np.errstate(divide='raise',over='ignore'): # 1/0.
        try:
            w = _nB(mu[1]-mu[0],beta)
        except FloatingPointError as e:
            if abs(f)<1e-18: # 0./0. -> 1.
                return 1.
            else:
                raise e
    return w * f

def _I2(A, B, epsA, epsB, mu, beta):
    f = A*B * np.real(
        _Psi(0,epsA,mu[1],beta) 
      - _Psi(0,epsA,mu[0],beta) 
      - _Psi(0,epsB,mu[1],beta)
      + _Psi(0,epsB,mu[0],beta)
    )
    with np.errstate(divide='raise',over='ignore'): # 1./0.
        try:
            w = _nB(mu[1]-mu[0],beta)
            dAB_inv = 1. / (epsA - epsB)
        except FloatingPointError as e:
            if abs(f)<1e-18: # 0./0.
                return 1.
            else:
                raise e
    return w * dAB_inv * f

#

Gamma1 = _I1


def Gamma2(A, B, epsA, epsB, mu, beta):
    return _I1(A,epsA,mu,beta) \
         + _I1(B,epsB,mu,beta) \
         + _I2(A,B,epsA,epsB,mu,beta)
