from hypothesis import given, settings, strategies as st
import pytest
from itertools import product
import torch, numpy as np
from neuromancer.psl.nonautonomous import systems
from neuromancer.psl.building_envelope import systems as building_systems
systems.update(building_systems)

from neuromancer.psl import pslgym
from neuromancer.psl.pslgym import PslGym
from stable_baselines3.common.env_checker import check_env

from neuromancer.psl.signals import sines

SEED=0

def get_name(emulator):
    return str(emulator)
    # if emulator in buildsys:
    # return emulator.keywords['system']
    # elif emulator in nonautosys:
    #     return str(emulator).split('.')[-1].split('\'')[0]
    # return None        

# systems = list(nonautosys)+list(buildsys)
# nonautosys = [v for v in nonautosys.values()]
# buildsys = [v for v in buildsys.values()]

# systems = nonautosys+buildsys
# systems = nonautosys

systems = { k:v for k,v in systems.items() if any((
           'TwoTank' in k,
           'SimpleSingleZone' in k,
           'Reno_full' in k,
           'HollandschHuys_full' in k))}

# @pytest.mark.parametrize("emulator", systems)
# @given(nsim=st.integers(3, 5))
# @settings(max_examples=1, deadline=None)
def test_pslgym(emulator, nsim):
    backend='numpy'
    env = PslGym(psl_sys=get_name(emulator), reward_instance=pslgym.MSE(max_r=.1), do_trace=True, act_momentum=0, seed=SEED, render_mode=None)
    check_env(env)
    system = systems[emulator](backend=backend, set_stats=True, seed=SEED)
    U = system.get_U(nsim+1, signal=sines)
    if 'TwoTank' in str(emulator):
        U[:,:] = U[0,:]
    obs, info = env.reset()
    y = env.obs_to_y(obs)
    assert(np.allclose(y, env.y))
    if env.has_d:
        sim = system.simulate(nsim=nsim, U=U, x0=env.X, D=env.D[:nsim])
    else:
        sim = system.simulate(nsim=nsim, U=U, x0=env.X)
    Y=sim['Y']
    # assert(sim['Dhidden'])
    for t in range(nsim):
        obs, reward, terminated, truncated, info = env.step(env.normalize_act(U[t,:]))
        y = env.obs_to_y(obs)
        assert(np.allclose(y, Y[t], rtol=1e-04, atol=1e-04))


if __name__=='__main__':
    for emulator in systems:
        print(emulator)
        test_pslgym(emulator, nsim=3)