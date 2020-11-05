from lookup import (
    get_spin_indices
)

def test_get_spin_indices():
    i = 5
    dup = 2
    ddw = 3
    iup, idw = get_spin_indices(i, dup, ddw)

    assert iup == i%dup
    assert idw == i//dup
