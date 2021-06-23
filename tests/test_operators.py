from hypothesis import given, settings, strategies as st
from neuromancer.operators import InterpolateAddMultiply
import torch


@given(st.lists(st.integers(1, 100),
                min_size=1, max_size=3))
@settings(max_examples=1000, deadline=None)
def test_interpolate_add_multiply_shape(shape):
    op = InterpolateAddMultiply()
    x = torch.randn(shape)
    y = torch.randn(shape)
    assert all([x.shape[i] == y.shape[i] for i in range(len(x.shape))])