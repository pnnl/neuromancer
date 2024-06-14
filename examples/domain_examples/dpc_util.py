from tqdm import tqdm
import random
import torch
from neuromancer.psl.signals import signals
import numpy as np
import os

# remove problematic signal
if 'beta' in signals:
  del signals['beta']

NSIM = 8929 - 3  # largest sim length allowed


def collect_stats(psl_sys):
    data = dict()
    for key in psl_sys.stats.keys():
        data[key] = []

    for i in range(10):
        U = psl_sys.get_U(nsim=NSIM + 1)
        sim = psl_sys.simulate(U=U, nsim=NSIM)
        for key in psl_sys.stats.keys():
            data[key].append(sim[key])

    stats = dict()
    for key in psl_sys.stats.keys():
        data[key] = np.concatenate(data[key], axis=0)
        for stat in 'min', 'max':
            stats[f'{key}_{stat}'] = torch.tensor(getattr(np, stat)(data[key], axis=0))

    return stats


def load_stats(psl_sys, psl_key):
    file_path = f'stats/{psl_key}.npz'
    stats = np.load(file_path, allow_pickle=True)
    stats = dict(stats)
    ret = {}
    for key, val in stats.items():
        for stat in ['min', 'max']:
            stat_value = getattr(psl_sys, f'u{stat}') if key == 'U' else val.item()[stat]
            ret[f'{key}_{stat}'] = torch.tensor(stat_value)
    return ret


def simulate(psl_sys, NV, nsim=20, nsteps=3500):
    data = {}
    for _ in tqdm(range(nsim), desc=f"Building data..."):
        nu = psl_sys.nU

        # halve starting X for building systems
        x0 = psl_sys.rng.uniform(low=NV['X_min'], high=(NV['X_max'] + NV['X_min']) / 2)

        signal_choice = random.choice(list(signals.keys()))

        signal = signals[signal_choice]

        ulow = NV['U_min']

        # halve maximum U for building systems, as random controls trend towards overheating
        uhigh = (NV['U_max'] + NV['U_min']) / 2
        U = signal(
            nsim=nsteps + 1,
            d=nu,
            min=ulow,
            max=uhigh,
        )

        sim_dict = psl_sys.simulate(nsim=nsteps, U=U, x0=x0)
        # make 3d for stacking simulations [sims, timesteps, dimensions]

        sim_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in sim_dict.items()}
        for key in sim_dict.keys():
            if key not in data:
                data[key] = sim_dict[key]
            else:
                data[key] = torch.cat([data[key], sim_dict[key]], dim=0)
    return data


def stack_refs(ref, forecast):
    ref = torch.tensor(ref, dtype=torch.float32)
    f_refs = [ref[:, :, 0:1]]
    for i in range(1, forecast):
        i_ref = torch.roll(ref, -i, dims=1)

        i_ref[:, -i:, :] = ref[:, -1:, :]
        f_refs.append(i_ref)

    return torch.cat(f_refs, dim=2)


def remove_key_prefix(d, prefix):
    new_d = {}
    for k, v in d.items():
        if k.startswith(prefix):
            new_d[k[len(prefix):]] = v

    return new_d