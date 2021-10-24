import numpy as np

from edpyt.fit_cg import Delta, DDelta

nmats = 3000
beta = 70.
nbath = 8
n = 2 * nbath

z = 1.j*(2*np.arange(nmats)+1)*np.pi/beta

delta = Delta(z)
ddelta = DDelta(z, n)

x = np.array([-1.98663495, -1.28445211, -0.42108352, -0.05721278,  0.05721278,  0.42108352,
     1.28445211,  1.98663495,  0.08997913,  0.18150846,  0.59537434,  0.31923968,
     0.31923968,  0.59537434,  0.18150846,  0.08997913])


c_delta = np.loadtxt('delta.txt')
c_ddelta = np.loadtxt('ddelta.txt')

py_delta = delta(x)
py_ddelta = ddelta(x)

