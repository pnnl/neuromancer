import os
from neuromancer import arg
import torch
import psl
from hypothesis import given, settings, strategies as st

gpu = '-gpu 0' if torch.cuda.is_available() else ''

systems = ([k for k, v in psl.datasets.items()
           if k not in ['vehicle3', '9bus_test']] +
          [k for k, v in psl.systems.items()
           if k not in ['UAV3D_kin', 'UAV2D_kin', 'vehicle3', 'UAV3D_reduced', 'InvPendulum']
           and not isinstance(v(), psl.emulator.ODE_Autonomous)])


p = arg.ArgParser(parents=[arg.policy(), arg.ctrl_loss(), arg.freeze()])
options = {k: v.choices for k, v in p._option_string_actions.items() if v.choices is not None and k != '-norm'}
args = []
for k, v in options.items():
    for opt in v:
        args.append(f'{k} {opt}')


@given(st.sampled_from(list(psl.datasets.keys()) + list(psl.emulators.keys())),
       st.sampled_from(args))
@settings(deadline=None, max_examples=20)
def test_opts(system, arg):
    os.system(f'python ../train_scripts/system_id.py -dataset {system} -epochs 1')
    code = os.system(f'python ../train_scripts/control.py -norm Y -savedir test_ctrl '
                     f'{arg} -epochs 1 -nsteps 8 -verbosity 1 -nsim 128 -system {system}')
    assert code == 0
