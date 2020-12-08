import numpy as np
import scipy
import time

from edpyt.matvec_product import (
    matvec_operator
)

def time_matvec_product():
    m = 100
    n = 200
    N = 1000
    Hup = scipy.sparse.random(m, m, density=0.3, format='csr')
    Hdw = scipy.sparse.random(n, n, density=0.3, format='csr')
    Hdd = np.random.random(m*n)
    vec = np.random.random(m*n)
    out = np.empty_like(vec)

    matvec = matvec_operator(Hdd, Hup, Hdw)
    s = time.time()
    for i in range(N):
        out = matvec(vec)
    e = time.time()
    return (e-s)/N

if __name__ == '__main__':
    elapsed = time_matvec_product()
    print('matvec_product:',elapsed)
