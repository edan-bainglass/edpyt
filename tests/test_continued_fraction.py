import numpy as np

from edpyt._continued_fraction import (
    continued_fraction
)


def test_continued_fraction():
    # Test pi 3.141592
    n = 500
    a = np.ones(n)*6
    b = 2.*np.arange(1,n+1) - 1
    b **= 2

    z = np.zeros(3, complex)
    c = continued_fraction(z, -a, -b)
    np.testing.assert_allclose(3.-c, np.pi)


def time_continued_fraction():
    from time import perf_counter
    # Test pi 3.141592
    n = 500
    a = np.ones(n)*6
    b = 2.*np.arange(1,n+1) - 1
    b **= 2

    z = np.zeros(3000, complex)
    start = perf_counter()
    for i in range(100):
        c = continued_fraction(z, -a, -b)
    elapsed = perf_counter() - start
    print('Elapsed : ', elapsed/100)
    np.testing.assert_allclose(3.-c, np.pi)

if __name__ == '__main__':
    time_continued_fraction()