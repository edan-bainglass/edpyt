import numpy as np

from edpyt.sector import (
    generate_states
)

def test_generate_states():

    states = generate_states(4, 2)
    expected = [
        int("0011",base=2),
        int("0101",base=2),
        int("0110",base=2),
        int("1001",base=2),
        int("1010",base=2),
        int("1100",base=2),
    ]
    np.testing.assert_allclose(states, expected)
