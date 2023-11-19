import dill
from mpi4py import MPI

MPI.pickle.__init__(dill.dumps, dill.loads)

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
