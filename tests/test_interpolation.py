import torch
from neuromancer.dynamics import interpolation
from hypothesis import given, settings, strategies as st


@given(st.integers(2, 30),
       st.integers(1, 5))
@settings(max_examples=500, deadline=None)
def test_LinInterp_Offline_dims(nsteps, nquerries):
    t = (torch.arange(nsteps) * 10).unsqueeze(-1)
    u = torch.sin(t)
    interp = interpolation.LinInterp_Offline(t, u)
    tq_vec = (torch.arange(nquerries) * 0.73 + 2).unsqueeze(-1)
    uq_vec = interp(tq_vec)

    assert uq_vec.shape[0] == nquerries
    assert uq_vec.shape[1] == u.shape[1]


@given(st.integers(2, 30),
       st.integers(1, 5),
       st.integers(1, 5))
@settings(max_examples=500, deadline=None)
def test_LinInterp_multi_u_Offline_dims(nsteps, nu, nquerries):
    t = (torch.arange(nsteps) * 10).unsqueeze(-1)
    u = torch.sin(t).repeat(1, nu)
    interp = interpolation.LinInterp_Offline(t, u)
    tq_vec = (torch.arange(nquerries) * 0.73 + 2).unsqueeze(-1)
    uq_vec = interp(tq_vec)
    assert uq_vec.shape[0] == nquerries
    assert uq_vec.shape[1] == u.shape[1]


@given(st.integers(1, 10),
       st.integers(1, 10))
@settings(max_examples=100, deadline=None)
def test_LinInterp_multi_u_Online_dims(nu, nbatches):
    t = torch.rand((nbatches, 2, 1))
    u = torch.rand((nbatches, 2, nu))
    tq = torch.rand((nbatches, 2, 1))
    interp = interpolation.LinInterp_Online()
    uq = interp(tq, t, u)
    assert uq.shape[0] == tq.shape[0]
    assert uq.shape[1] == u.shape[2]

# %%
