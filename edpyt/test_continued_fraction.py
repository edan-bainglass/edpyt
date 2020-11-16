import numpy as np

from gf_lanczos import (
    continued_fraction
)


def test_continued_fraction():
    # Test pi 3.141592
    n = 500
    a = np.ones(n)*6
    b = 2.*np.arange(1,n+1) - 1
    b **= 2

    cf = continued_fraction(-a, -b)
    np.testing.assert_allclose(3.-cf(0.,0.), np.pi)
