from hypothesis import given, settings, strategies as st
import pytest
from neuromancer.psl.signals import signals as sig, nd_walk
import numpy as np

signals = [v for k, v in sig.items()]
bounds = [(0., 1.), (-1., 0.), (1., 2.), (-2., -1.), (-1., 1.)]

SEED = 0
rng = np.random.default_rng(seed=SEED)

@pytest.mark.parametrize("signal", signals)
@given(nsim=st.integers(3, 5), d=st.integers(1, 3))
def test_signal_shape(signal, nsim, d):
    """
    All signals should be nsim x d np.arrays of type np.float64
    """
    out = signal(nsim, d)
    assert isinstance(out, np.ndarray)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.shape == (nsim, d)


@given(signal=st.sampled_from(signals), nsim=st.integers(3, 5), d=st.integers(1, 3), bound=st.sampled_from(bounds))
def test_signal_bound(signal, nsim, d, bound):
    """
    All signals bounded by dimensionwise min and max should be bounded correctly
    """
    kwargs = {"min": bound[0], "max": bound[1]}
    out = signal(nsim, d, **kwargs, rng=rng)
    assert not np.isnan(out).any()
    assert out.max() <= bound[1]
    assert out.min() >= bound[0]


@pytest.mark.parametrize("avoid_edge", [True, False])
def test_nd_walk_picks_start_when_x0_is_none(avoid_edge):
    """
    nd_walk should pick its own start for either edge setting. nd_walk is not
    in the signals registry, so the parametrized tests above never reach it.
    """
    out = nd_walk(50, 3, min=-2., max=1.5, x0=None, avoid_edge=avoid_edge, rng=rng)
    assert out.shape == (50, 3)
    assert not np.isnan(out).any()
    assert out.min() >= -2. and out.max() <= 1.5

