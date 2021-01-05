from edpyt.espace import adjust_neigsector

import numpy as np
from dataclasses import dataclass
from edpyt.sector import get_sector_dim
from edpyt.lookup import get_sector_index


@dataclass
class DummySector:
    eigvals: np.ndarray = np.empty(3)


n = 6
keys = [(1,2),(3,4),(5,6)] # dims = [90, 300, 6]
iscts = [get_sector_index(n,nup,ndw) for nup, ndw in keys]
espace = dict.fromkeys(keys, DummySector())

neig = np.ones((n+1)*(n+1),int) * 3
neig[iscts[0]] += 1 # isct[0] has less eigen-states than neig -> DECREASE neig!

neig_expected = neig.copy()
neig_expected[iscts[0]] -= 1

neig_expected[np.setdiff1d(
    np.arange((n+1)*(n+1)),
    iscts)] -= 1
neig_expected[iscts[1:]] += 1 # isct[1:] has eq. eigen-states as neig -> INCREASE neig!

def test_adjust_neigsector():
    adjust_neigsector(espace, neig, n)
    np.testing.assert_allclose(neig, neig_expected)
