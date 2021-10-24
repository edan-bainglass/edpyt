import numpy as np
from edpyt.fit_cg import get_initial_bath


def test_initial_bath():
    # ODD
    nbath = 5
    p = get_initial_bath(nbath=nbath)
    np.testing.assert_allclose(p[:nbath], [-2.,-1.,0.,1.,2.])
    np.testing.assert_allclose(p[nbath:], 1/np.sqrt(nbath))
    # EVEN
    nbath = 6
    p = get_initial_bath(nbath=nbath)
    np.testing.assert_allclose(p[:nbath], [-2.,-1.,-0.1,0.1,1.,2.])
    np.testing.assert_allclose(p[nbath:], 1/np.sqrt(nbath))
