import numpy as np
from mpi4py import MPI
  

def collect_dw(a, comm=None):
    """Transpose a hilbert vector.
    
    NOTE : Works only for sector with equal up and down # of spins.

P0:
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]]
P1:
    [[18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 31 32 33 34 35]]
P0:
    [[ 0  1  2]
     [ 6  7  8]
     [12 13 14]
     [18 19 20]
     [24 25 26]
     [30 31 32]]
P1:
    [[ 3  4  5]
     [ 9 10 11]
     [15 16 17]
     [21 22 23]
     [27 28 29]
     [33 34 35]]

    """

    m, n = a.shape
    
    if comm is None:
        comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert m*size==n, "Non-equal # of up and down spins,"
    
    batch = n//size
    chunk = batch*m

    b = np.empty(a.T.shape,a.dtype)
    for p in range(size):
        b[p*batch:(p+1)*batch,:] = a[:,p*batch:(p+1)*batch]

    comm.Alltoall([b,chunk], [a,chunk])

    return a.reshape(n,m)


def collect_up(a, comm=None):
    """Transpose a hilbert vector.
    
    NOTE : Works only for sector with equal up and down # of spins.

P0:
    [[ 0  1  2]
     [ 6  7  8]
     [12 13 14]
     [18 19 20]
     [24 25 26]
     [30 31 32]]
P1:
    [[ 3  4  5]
     [ 9 10 11]
     [15 16 17]
     [21 22 23]
     [27 28 29]
     [33 34 35]]
P0:
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]]
P1:
    [[18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 31 32 33 34 35]]
    
    """

    m, n = a.shape
    # assert m==n, "Non-equal # of up and down spins,"
    
    if comm is None:
        comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert m==n*size, "Non-equal # of up and down spins,"
    
    batch = m//size
    chunk = batch*n

    b = np.empty(a.shape,a.dtype)

    comm.Alltoall([a,chunk], [b,chunk])
    b.shape = (size,-1)
    a.shape = (n,m)
    for p in range(batch):
        a[p,:] = b[:,p*n:(p+1)*n].flat

    return a