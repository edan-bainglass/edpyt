import numpy as np
from mpmath import psi as polygamma
from scipy.constants import hbar as _hbar, eV
# from scipy.special import polygamma

hbar = _hbar/eV # Units of electron volt.

# https://www.weizmann.ac.il/condmat/oreg/sites/condmat.oreg/files/uploads/Thesises/carmiphd.pdf

# def Psi(n, a, b, beta):
#     """Polygamma function of order n."""
#     return psi(n, 0.5 + (1.j*beta)/(2.*np.pi)*(a-b))

    
def Psi(n, a, b, beta):
    return polygamma(n, 0.5 + (1.j*beta)/(2.*np.pi)*(a-b))


def Taylor(n, x, a, b, beta):
    k = Psi(n, a, b, beta)
    return k / x \
           - 0.5 * k \
           + 1/12 * x * k \
           - 1/720 * x**3 * k

# fmax = 1e15
# fmin = 1/fmax

# def nB(eps, beta):
#     """Bose distribution."""
#     return 1./(np.exp(beta*eps)-1)

nB = lambda x: 1./(np.exp(x)-1)


def I1(C, eps, mu, beta):
    x = (mu[1]-mu[0]) * beta
    if abs(x) < 1e-16:
        return 0. #np.sign(x)
    if abs(x) < 1e-15:
        return C**2 * beta/(2*np.pi) * np.imag(
            Taylor(1,x,mu[1],eps,beta)
          - Taylor(1,x,mu[0],eps,beta)
        )        
    return C**2 * beta/(2*np.pi) * nB(x) * np.imag(
            Psi(1,mu[1],eps,beta)
          - Psi(1,mu[0],eps,beta)
        )
    # f = C**2 * beta/(2*np.pi) * np.imag(
    #     Psi(1,mu[1],eps,beta) 
    #   - Psi(1,mu[0],eps,beta)
    # )
    # with np.errstate(divide='raise',over='ignore'): # 1/0.
    #     try:
    #         w = nB(mu[1]-mu[0],beta)
    #     except FloatingPointError as e: # nB -> oo
    #         if abs(f)<1e-13: # oo.*0. -> 0.
    #             return 1#0.
    #         else:
    #             raise e
    # return w * f

def I2(A, B, epsA, epsB, mu, beta):
    x = (mu[1]-mu[0])*beta
    if abs(x) < 1e-16:
        return 0. # -1.
    if abs(x) < 1e-15:
        return A * B / (epsA-epsB) * np.real(
              Taylor(0,epsA,mu[1],beta) 
            - Taylor(0,epsA,mu[0],beta) 
            - Taylor(0,epsB,mu[1],beta)
            + Taylor(0,epsB,mu[0],beta)
        )
    return A * B * nB(x) / (epsA-epsB) * np.real(
          Psi(0,epsA,mu[1],beta) 
        - Psi(0,epsA,mu[0],beta) 
        - Psi(0,epsB,mu[1],beta)
        + Psi(0,epsB,mu[0],beta)
    )
    # f = A*B * np.real(
    #     Psi(0,epsA,mu[1],beta) 
    #   - Psi(0,epsA,mu[0],beta) 
    #   - Psi(0,epsB,mu[1],beta)
    #   + Psi(0,epsB,mu[0],beta)
    # )
    # with np.errstate(divide='raise',over='ignore'): # 1./0.
    #     try:
    #         w = nB(mu[1]-mu[0],beta)
    #         dAB_inv = 1. / (epsA - epsB)
    #     except FloatingPointError as e: # nB -> oo or 1/dE -> oo
    #         if abs(f)<1e-13: # oo*0. -> 0.
    #             return 1#0.
    #         else:
    #             raise e
    # return w * dAB_inv * f

#

Gamma1 = I1


def Gamma2(A, B, epsA, epsB, mu, beta):
    return I1(A,epsA,mu,beta) \
         + I1(B,epsB,mu,beta) \
         + I2(A,B,epsA,epsB,mu,beta)


def Gamma4(A, B, C, D, epsA, epsB, epsC, epsD, mu, beta):
    return Gamma2(A,B,epsA,epsB,mu,beta) \
         + Gamma2(A,C,epsA,epsC,mu,beta) \
         - Gamma2(A,-D,epsA,epsD,mu,beta) \
         - Gamma2(B,-C,epsB,epsC,mu,beta) \
         + Gamma2(B,D,epsB,epsD,mu,beta) \
         + Gamma2(C,D,epsC,epsD,mu,beta)