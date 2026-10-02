"""
Benchmark for the System rollout.

Measures how `System.forward` scales with the rollout horizon, in three ways:

  time       wall-clock for the rollout, and for rollout + backward
  allocated  total bytes the rollout asks the allocator for, whether or not it keeps them
  peak RSS   high-water resident memory of a rollout + backward

The horizons double, so each column's scaling can be read straight off the table. Allocated
bytes are worth watching independently of peak RSS -- memory that is freed immediately still
costs time to zero and copy, and an allocator handed a long run of ever-larger requests cannot
reuse the blocks it just freed, so churn shows up as resident memory even though nothing is
retained.

Nothing here reaches into the rollout's internals, so the same command is comparable across
changes to how the rollout accumulates its output.

Usage::

    python benchmarks/bench_rollout.py                    # System, default horizons
    python benchmarks/bench_rollout.py --preview          # SystemPreview instead
    python benchmarks/bench_rollout.py --nsteps 100 200 400 --batch 64
    python benchmarks/bench_rollout.py --no-memory        # timings only, much faster
"""
import argparse
import multiprocessing
import resource
import time

import torch
from torch.profiler import ProfilerActivity, profile

from neuromancer.modules import blocks
from neuromancer.system import Node, System, SystemPreview

# Single-zone building model dimensions, matching examples/domain_examples/building_control.py
NX, NU, ND, NY = 4, 1, 3, 1


def build_system(nsteps, preview=False):
    """
    Closed-loop building control system: disturbance observer -> neural policy -> linear state
    space model -> output map, with the state fed back to the policy.

    :param nsteps: (int) rollout horizon
    :param preview: (bool) give the policy a past/future window of the comfort bounds, which
                    exercises SystemPreview's windowed reads instead of single-step reads
    :return: (System) closed-loop system
    """
    torch.manual_seed(0)
    A = torch.eye(NX) + 0.01 * torch.randn(NX, NX)
    # Scale to a spectral radius below 1. Left unstable, the closed-loop state reaches 1e13 by
    # step 2000 and overflows to inf/nan by step 8000, which makes long horizons meaningless.
    A = 0.99 * A / torch.linalg.eigvals(A).abs().max()
    B = 0.1 * torch.randn(NX, NU)
    C = torch.randn(NY, NX)
    E = 0.1 * torch.randn(NX, ND)

    window = {'past': 2, 'future': 2}
    input_map = {'ymin': dict(window), 'ymax': dict(window)} if preview else None
    span = window['past'] + 1 + window['future'] if preview else 1

    net = blocks.MLP_bounds(insize=NY + 2 * NY * span + 2, outsize=NU, hsizes=[32, 32],
                            min=torch.tensor([0.]), max=torch.tensor([5000.]))
    nodes = [
        Node(lambda d: d[:, :2], ['d'], ['d_obsv'], name='dist_obsv'),
        Node(net, ['y', 'ymin', 'ymax', 'd_obsv'], ['u'], name='policy', input_map=input_map),
        Node(lambda x, u, d: x @ A.T + u @ B.T + d @ E.T, ['x', 'u', 'd'], ['x'], name='SSM'),
        Node(lambda x: x @ C.T, ['x'], ['y'], name='y=Cx'),
    ]
    system_class = SystemPreview if preview else System
    return system_class(nodes, nsteps=nsteps, name='cl_system')


def make_data(nsteps, batch):
    """
    One batch of training data: initial state, comfort bounds and disturbance trajectories.
    """
    torch.manual_seed(1)
    x0 = torch.randn(batch, 1, NX)
    ymin = 18.0 + torch.rand(batch, 1, NY) * torch.ones(batch, nsteps + 1, NY)
    return {'x': x0, 'y': x0[:, :, [0]], 'ymin': ymin, 'ymax': ymin + 2.0,
            'd': torch.randn(batch, nsteps + 1, ND)}


def best_of(run, rounds=5, warmup=1):
    """
    Fastest of several timed runs. The minimum rather than the mean, because the noise here is
    scheduler and allocator interference, which only ever adds time.

    :param run: (callable) zero-argument callable to time
    :return: (float) best wall-clock seconds observed
    """
    for _ in range(warmup):
        run()
    return min(_timed(run) for _ in range(rounds))


def _timed(run):
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


def bytes_allocated(run):
    """
    Total CPU bytes requested during run, summed over every allocating operation.

    This is turnover, not occupancy: a rollout that builds its output by repeatedly growing
    one tensor allocates far more than it ends up holding.

    :param run: (callable) zero-argument callable to profile
    :return: (int) bytes allocated
    """
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        run()
    return sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0)


def rollout_and_backward(system, data):
    """
    One rollout followed by a backward pass over it -- what a training step actually pays.

    Autograd has to start from a scalar, and a rollout returns a dict of trajectories, so one
    of them gets reduced. Which one is arbitrary: every output traces back through the whole
    recurrence, so any choice walks the same graph. Substituting the mean, or an output other
    than the control, or the sum of all of them, moves the measured cost by under half a
    percent. Squared mean control effort is used because that is what a DPC objective
    penalises, so the shape of the backward pass matches a real one.

    Deliberately not a real Problem/PenaltyLoss: this benchmark isolates the rollout, and
    building the loss machinery would put that on the clock too.

    :param system: (System) closed-loop system to roll out
    :param data: (dict {str: Tensor}) rollout inputs
    """
    system(data)['u'].square().mean().backward()


def peak_rss_growth(nsteps, batch, preview):
    """
    Peak resident memory a rollout + backward adds on top of everything already loaded.

    Reported as growth because the absolute figure is mostly the torch import. ru_maxrss is a
    high-water mark that never falls, so this is only meaningful once per process: measured
    in-process after a larger horizon it reads zero. run spawns a fresh process per call.

    :return: (float) megabytes
    """
    system = build_system(nsteps, preview)
    data = make_data(nsteps, batch)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rollout_and_backward(system, data)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (after - before) / (1024 * 1024)


def measure(nsteps, batch, preview):
    """
    Time and allocation for one horizon.

    :return: (dict) forward seconds, forward+backward seconds, bytes allocated by the forward
    """
    system = build_system(nsteps, preview)
    data = make_data(nsteps, batch)

    def forward():
        with torch.no_grad():
            system(data)

    def forward_backward():
        rollout_and_backward(system, data)

    return {'fwd': best_of(forward),
            'fwd_bwd': best_of(forward_backward, rounds=3),
            'allocated': bytes_allocated(forward)}


def run(horizons, batch, preview, with_memory):
    """Measures every horizon and prints the table."""
    rows = [measure(n, batch, preview) for n in horizons]
    if with_memory:
        spawn = multiprocessing.get_context('spawn')
        for nsteps, row in zip(horizons, rows):
            # one fresh process per horizon, see peak_rss_growth
            with spawn.Pool(1, initializer=torch.set_num_threads, initargs=(1,)) as pool:
                row['rss'] = pool.apply(peak_rss_growth, (nsteps, batch, preview))

    label = 'SystemPreview' if preview else 'System'
    print(f'\n{label}.forward, batch={batch}\n')
    header = f"{'nsteps':>7}{'fwd ms':>12}{'fwd+bwd ms':>14}{'allocated MB':>16}"
    if with_memory:
        header += f"{'peak RSS MB':>14}"
    print(header)
    print('-' * len(header))
    for nsteps, row in zip(horizons, rows):
        line = (f"{nsteps:>7}{row['fwd'] * 1e3:>12.2f}{row['fwd_bwd'] * 1e3:>14.2f}"
                f"{row['allocated'] / 1e6:>16.1f}")
        if with_memory:
            line += f"{row['rss']:>14.1f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsteps', nargs='+', type=int, default=[250, 500, 1000, 2000],
                        help='rollout horizons; doubling them makes the scaling readable')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--preview', action='store_true',
                        help='benchmark SystemPreview instead of System')
    parser.add_argument('--no-memory', action='store_true',
                        help='skip peak RSS, which spawns a process per horizon')
    args = parser.parse_args()

    torch.set_num_threads(1)
    run(args.nsteps, args.batch, args.preview, not args.no_memory)


if __name__ == '__main__':
    main()
