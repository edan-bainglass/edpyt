import numpy as np
from scipy.sparse import kronsum, random
from time import perf_counter

from edpyt.ham_hopping import DwHopping, UpHopping
from edpyt.ham_local import Local
from edpyt.matvec_product import matvec_operator, todense


def test_matvec_product():

    dup = 10
    dwn = 20

    Hup = UpHopping(random(dup,dup,density=0.3,format='csr'),dwn)
    Hdw = DwHopping(random(dwn,dwn,density=0.3,format='csr'),dup)
    Hdd = np.random.random(dup*dwn).view(Local)

    H = todense(Hdd, Hup, Hdw)

    vec = np.random.random(dup*dwn)

    sp_matvec = matvec_operator(Hdd, Hup, Hdw)
    s = perf_counter()
    res = sp_matvec(vec)
    e = perf_counter()
    print(e-s)
    assert np.allclose(H.dot(vec), res)
    

def time_kronsum():

    dup = 10
    dwn = 20
    Hup = random(dup,dup,density=0.1)
    Hdw = random(dwn,dwn,density=0.1)

    nrep = 20

    elapsed = 0.
    for i in range(nrep):
        s = perf_counter()
        a = kronsum(Hup,Hdw).todense()
        e = perf_counter()
        elapsed += e - s
    print(f'f.__name__: {elapsed/nrep}')
    
    elapsed = 0.
    for i in range(nrep):
        s = perf_counter()
        b = np.kron(np.eye(dwn),Hup.todense()) + np.kron(Hdw.todense(),np.eye(dup))
        e = perf_counter()
        elapsed += e - s
    print(f'f.__name__: {elapsed/nrep}')

    np.testing.assert_allclose(a,b)


if __name__ == '__main__':
    time_kronsum()