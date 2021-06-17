import os
from neuromancer import arg
from neuromancer import datasets
import torch

gpu = '-gpu 0' if torch.cuda.is_available() else ''


def test_opts():
    p = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(), arg.loss(), arg.lin(), arg.ssm()])
    options = {k: v.choices for k, v in p._option_string_actions.items() if v.choices is not None and k != '-norm'}
    systems = list(datasets.systems.keys())
    results = dict()
    for i, (k, v) in enumerate(options.items()):
        for j, opt in enumerate(v):
            key = '_'.join([k, opt, systems[(i*j) % len(systems)]])
            print(k, opt, systems[(i*j) % len(systems)])
            code = os.system(f'python ../train_scripts/system_id.py -norm Y '
                             f'{k} {opt} -epochs 1 -nsteps 8 -verbosity 1 -nsim 128 -system {systems[(i*j) % len(systems)]}')
            results[key] = code

    failures = {k: v for k, v in results.items() if v != 0}
    assert not failures, f'Test failures: {failures}'


def test_opts_constraints():
    p = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(), arg.loss(), arg.lin(), arg.ssm()])
    options = {k: v.choices for k, v in p._option_string_actions.items() if v.choices is not None and k != '-norm'}
    systems = list(datasets.systems.keys())
    results = dict()
    for i, (k, v) in enumerate(options.items()):
        for j, opt in enumerate(v):
            key = '_'.join([k, opt, systems[(i*j) % len(systems)]])
            print(k, opt, systems[(i*j) % len(systems)])
            code = os.system(f'python ../train_scripts/system_id_constraints.py -norm Y '
                             f'{k} {opt} -epochs 1 -nsteps 8 -verbosity 1 -nsim 128 -system {systems[(i*j) % len(systems)]}')
            results[key] = code

    failures = {k: v for k, v in results.items() if v != 0}
    assert not failures, f'Test failures: {failures}'


if __name__ == '__main__':
    test_opts_constraints()