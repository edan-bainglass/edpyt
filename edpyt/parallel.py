import os
import sys


def paropen(name):
    """MPI-safe version of open function.

    The file is opened on the master only, and /dev/null
    is opened on all other nodes.
    """
    if comm.rank > 0:
        name = os.devnull
    return open(name, 'w', 1)


def parprint(*args, **kwargs):
    """MPI-safe print - prints only from master. """
    if comm.rank == 0:
        print(*args, **kwargs)


class SerialComm:
    size = 1
    rank = 0


class ParallelComm:
    def __init__(self, comm=None) -> None:
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

    @property
    def size(self):
        return self.comm.size

    @property
    def rank(self):
        return self.comm.rank

    def __getattr__(self, name: str):
        return getattr(self.comm, name)


# mpi_dtype = {
#     'd':'DOUBLE',
#     'f':'FLOAT',
#     'F':'COMPLEX8',
#     'D':'COMPLEX16'
# }

comm = None

if 'mpi4py' in sys.modules:
    comm = ParallelComm()
else:
    comm = SerialComm()

