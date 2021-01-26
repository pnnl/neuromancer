import os
from neuromancer import arg
import argparse
from neuromancer import datasets

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=None)
args = parser.parse_args()
if args.gpu is not None:
    gpu = f'-gpu {args.gpu}'

p = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(), arg.loss(), arg.lin(), arg.ssm()])
options = {k: v.choices for k, v in p._option_string_actions.items() if v.choices is not None and k != '-norm'}
systems = list(datasets.systems.keys())
for i, (k, v) in enumerate(options.items()):
    for j, opt in enumerate(v):
        print(k, opt)
        code = os.system(f'python ../train_scripts/system_id.py -norm Y '
                         f'{k} {opt} -epochs 1 -nsteps 8 -verbosity 1 -nsim 128 -system {systems[(i*j) % len(systems)]}')
        assert code == 0, f'Failure on flag {k} with value {opt}.'