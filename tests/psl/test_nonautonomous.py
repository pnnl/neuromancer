from neuromancer.psl.nonautonomous import CSTR


def test_CSTR_instantiation():
    """
    Test instantiation of CSTR
    """
    modelSystem = CSTR()
    modelSystem.simulate(nsim=2000)
