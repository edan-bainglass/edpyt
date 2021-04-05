from edpyt.integrate_gf import integrate_gf

def test_integrate_gf():
    poles = [10,5,2,-5]
    gf = lambda z: sum(1/(z+p) for p in poles)
    computed = integrate_gf(gf)
    assert abs(3-computed)<1e-4