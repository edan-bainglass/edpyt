import sys

from edpyt.mpi import RANK


def pprint(msg, end="\n"):
    if RANK == 0:
        print(msg, end=end)
        sys.stdout.flush()
