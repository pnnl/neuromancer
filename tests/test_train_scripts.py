import os
from deepmpc.train_scripts import system_id
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=None)
args = parser.parse_args()
if args.gpu is not None:
    gpu = f'-gpu {args.gpu}'



p = system_id.parse()

options = {k: v.choices for k, v in p._option_string_actions.items() if v.choices is not None and k != '-norm'}

for k, v in options.items():
    for opt in v:
        print(k, opt)
        os.system(f'source activate mpc; python ../deepmpc/train_scripts/system_id.py -norm Y '
                  f'{k} {opt} -epochs 1 -nsteps 2 -verbosity 1 -norm YU')
        exit()
