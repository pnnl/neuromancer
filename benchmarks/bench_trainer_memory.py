"""
Benchmark for the Trainer's memory footprint over a training run.

Measures how a `Trainer.train()` call scales with the number of epochs, in four ways:

  time       wall-clock for the whole run, and per epoch
  footprint  memory the process holds after the run, on top of what it held before the trainer
             existed, counting pages the OS has compressed or swapped out
  held       bytes of tensor storage reachable from the trainer object, through Python
             references and through the autograd graphs of any tensor it stores
  nodes      autograd graph nodes reachable the same way; a graph whose backward pass has
             already run holds no tensors, but its nodes are still on the books

A trainer keeps per-epoch bookkeeping (loss history, best dev loss, best weights) for the
whole run, so some retained memory is expected. What is not expected is retained memory that
grows with the number of epochs: nothing the trainer needs to remember about an epoch is
bigger than a few scalars and one copy of the weights. The epochs double, so growth can be read
straight off the table -- a footprint that doubles with the epochs is a leak.

The dev evaluation is run twice, with and without `grad_inference`, because that is what
decides whether a dev loss carries an autograd graph behind it. Anything the trainer stores
without detaching drags that graph, and every activation it saved, along for the whole run.

The two memory figures answer different questions. Footprint is what the machine pays, but
the allocator and the OS sit between it and the trainer: freed blocks are cached, and idle
pages get compressed, so it is noisy and lags -- and, since the epoch counts in a sweep share
one process (see run), that noise compounds across a row's trials rather than resetting for
each one. Held is exact, and it is attributed: it counts what the trainer itself is keeping
alive, which is what a fix to the trainer can change, and process reuse cannot affect it.

Nothing here reaches into the trainer's internals -- the walk starts from the trainer object
and follows whatever it references -- so the same command is comparable across changes to what
the trainer keeps.

Usage::

    python benchmarks/bench_trainer_memory.py                       # both modes, default epochs
    python benchmarks/bench_trainer_memory.py --epochs 100 200 400
    python benchmarks/bench_trainer_memory.py --grad-inference off  # dev loss without a graph
    python benchmarks/bench_trainer_memory.py --hsize 512 --layers 6  # bigger weights to snapshot
"""
import argparse
import contextlib
import ctypes
import gc
import io
import multiprocessing
import os
import sys
import time
import types

import torch
import torch.nn as nn

from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer


def build_problem(hsize, layers, grad_inference):
    """
    Parametric Rosenbrock problem, matching tests/test_trainer.py: an MLP maps the parameters
    (a, p) to a solution x, penalised by the objective and three inequality constraints.

    :param hsize: (int) hidden layer width
    :param layers: (int) number of hidden layers
    :param grad_inference: (bool) keep the autograd graph during dev evaluation
    :return: (Problem)
    """
    torch.manual_seed(0)
    func = blocks.MLP(insize=2, outsize=2, bias=True, nonlin=nn.ReLU, hsizes=[hsize] * layers)
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')

    x1 = variable('x')[:, [0]]
    x2 = variable('x')[:, [1]]
    p = variable('p')
    a = variable('a')

    obj = ((1 - x1) ** 2 + a * (x2 - x1 ** 2) ** 2).minimize(weight=1.0, name='obj')
    Q_con = 100.
    constraints = [Q_con * (x1 >= x2),
                   Q_con * ((p / 2) ** 2 <= x1 ** 2 + x2 ** 2),
                   Q_con * (x1 ** 2 + x2 ** 2 <= p ** 2)]
    loss = PenaltyLoss([obj], constraints)
    return Problem([sol_map], loss, grad_inference=grad_inference)


def make_loaders(nsim, batch):
    """
    Train and dev loaders over uniformly sampled problem parameters.

    :param nsim: (int) samples per split
    :param batch: (int) batch size
    :return: (DataLoader, DataLoader)
    """
    torch.manual_seed(1)
    loaders = []
    for name in ['train', 'dev']:
        samples = {'a': torch.FloatTensor(nsim, 1).uniform_(0.2, 1.2),
                   'p': torch.FloatTensor(nsim, 1).uniform_(0.5, 2.0)}
        data = DictDataset(samples, name=name)
        loaders.append(torch.utils.data.DataLoader(data, batch_size=batch, num_workers=0,
                                                   collate_fn=data.collate_fn, shuffle=False))
    return loaders


def footprint():
    """
    Memory this process holds right now, in bytes, including what the OS has compressed or
    paged out.

    Resident size alone would not do: once a process reaches a few gigabytes the OS starts
    compressing or swapping the pages it is not touching, and its RSS stops growing, or falls,
    while the process keeps allocating. Each platform has a figure that keeps counting:

      macOS    physical footprint, what Activity Monitor reports
      Linux    VmRSS + VmSwap from /proc/self/status
      Windows  private bytes, the commit charge that Task Manager reports
    """
    if sys.platform == 'darwin':
        return _darwin_phys_footprint()
    if sys.platform == 'win32':
        return _windows_private_bytes()
    return _linux_rss_plus_swap()


class _RusageInfoV0(ctypes.Structure):
    """struct rusage_info_v0 from <sys/resource.h>."""
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16),
                ('ri_user_time', ctypes.c_uint64),
                ('ri_system_time', ctypes.c_uint64),
                ('ri_pkg_idle_wkups', ctypes.c_uint64),
                ('ri_interrupt_wkups', ctypes.c_uint64),
                ('ri_pageins', ctypes.c_uint64),
                ('ri_wired_size', ctypes.c_uint64),
                ('ri_resident_size', ctypes.c_uint64),
                ('ri_phys_footprint', ctypes.c_uint64),
                ('ri_proc_start_abstime', ctypes.c_uint64),
                ('ri_proc_exit_abstime', ctypes.c_uint64)]


def _darwin_phys_footprint():
    info = _RusageInfoV0()
    libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
    if libproc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
        raise OSError('proc_pid_rusage failed')
    return info.ri_phys_footprint


class _ProcessMemoryCountersEx(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX from <psapi.h>."""
    _fields_ = [('cb', ctypes.c_uint32),
                ('PageFaultCount', ctypes.c_uint32),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t)]


def _windows_private_bytes():
    info = _ProcessMemoryCountersEx()
    info.cb = ctypes.sizeof(info)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(info), info.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return info.PrivateUsage


def _linux_rss_plus_swap():
    total = 0
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith(('VmRSS:', 'VmSwap:')):
                total += int(line.split()[1]) * 1024  # reported in kB
    return total


def held(root):
    """
    Tensor storage and autograd graph reachable from root.

    Follows Python references through containers and instance dicts, and autograd references
    from any tensor with a grad_fn: the graph behind it, and the tensors each node saved for
    its backward pass. Storage is counted once however many tensors or nodes view it.

    Functions, classes and modules are not followed, since every one of them leads back to the
    torch namespace and from there to everything.

    :param root: (object) where to start
    :return: (int, int) bytes of tensor storage, autograd nodes
    """
    total, nodes = 0, 0
    seen_objects, seen_storage = set(), set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))

        if torch.is_tensor(obj):
            storage = obj.untyped_storage()
            if storage.data_ptr() not in seen_storage:
                seen_storage.add(storage.data_ptr())
                total += storage.nbytes()
            if obj.grad_fn is not None:
                stack.append(obj.grad_fn)
        elif isinstance(obj, torch.autograd.graph.Node):
            nodes += 1
            for name in dir(obj):
                if name.startswith('_saved_'):
                    try:
                        stack.append(getattr(obj, name))
                    except RuntimeError:
                        pass  # freed by a backward pass: nothing held
            stack.extend(fn for fn, _ in obj.next_functions if fn is not None)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, (type, types.ModuleType,
                                                               types.FunctionType,
                                                               types.MethodType)):
            stack.append(vars(obj))
    return total, nodes


def measure(epochs, hsize, layers, nsim, batch, grad_inference):
    """
    One full training run, in a process of its own.

    Footprint is reported as growth over the state before the trainer existed, so the torch
    import and the data are not on the books. Every run gets a fresh process, so that the
    allocator's cache from one run is not mistaken for memory held by the next.

    :return: (dict) seconds for the run, footprint growth in bytes, bytes and autograd nodes
             held by the trainer
    """
    torch.set_num_threads(1)
    problem = build_problem(hsize, layers, grad_inference)
    train_loader, dev_loader = make_loaders(nsim, batch)
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)

    gc.collect()
    footprint_before = footprint()

    # patience and warmup beyond the horizon, so every run trains for exactly `epochs`
    trainer = Trainer(problem, train_loader, dev_loader, optimizer=optimizer, epochs=epochs,
                      patience=epochs + 1, warmup=epochs + 1, epoch_verbose=epochs + 1)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # the trainer prints epoch 0 regardless
        trainer.train()
    seconds = time.perf_counter() - start

    gc.collect()
    held_bytes, nodes = held(trainer)
    return {'seconds': seconds,
            'footprint': footprint() - footprint_before,
            'held': held_bytes,
            'nodes': nodes}


def run(epoch_counts, hsize, layers, nsim, batch, grad_inference):
    """
    Measures every epoch count, printing each row as it lands.

    Every epoch count in the sweep shares one process, rather than each getting its own: torch's
    import is the dominant cost at the epoch counts this is normally run at (a fresh interpreter
    takes seconds just to import torch and neuromancer, before a single batch runs), so it is
    worth paying exactly once per grad_inference mode instead of once per epoch count.

    That reuse costs footprint specifically. Once one trial has faulted in and freed some
    memory, the allocator keeps the freed blocks rather than returning them to the OS, so a
    later trial's allocations can be served from that cache without the process's footprint
    growing to show it -- the same effect that makes footprint noisy in the first place (see
    the module docstring), now compounding trial to trial instead of resetting for each one.
    held and nodes are unaffected: they are an exact walk over live references, not a reading
    of a live OS counter, so process history cannot hide anything from them. Where footprint's
    absolute number matters, run that one epoch count on its own -- a lone invocation is still
    a fresh process, so its one trial is uncontaminated.

    A run at the larger epoch counts takes minutes, so the header and prior rows are printed
    right away rather than held back until the whole sweep finishes -- otherwise a long run
    looks identical to a hang.
    """
    print(f'\nTrainer.train, grad_inference={grad_inference}, MLP {layers}x{hsize}, '
          f'{nsim} samples, batch={batch}\n')
    header = (f"{'epochs':>7}{'train s':>10}{'ms/epoch':>10}{'footprint MB':>14}{'held MB':>10}"
              f"{'nodes':>10}")
    print(header, flush=True)
    print('-' * len(header), flush=True)

    spawn = multiprocessing.get_context('spawn')
    with spawn.Pool(1) as pool:  # one process, reused for every epoch count in this sweep
        for epochs in epoch_counts:
            row = pool.apply(measure, (epochs, hsize, layers, nsim, batch, grad_inference))
            print(f"{epochs:>7}{row['seconds']:>10.1f}{row['seconds'] / epochs * 1e3:>10.1f}"
                  f"{row['footprint'] / 1e6:>14.1f}{row['held'] / 1e6:>10.1f}{row['nodes']:>10}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', nargs='+', type=int, default=[100, 200, 400],
                        help='epoch counts; doubling them makes the scaling readable')
    parser.add_argument('--hsize', type=int, default=256, help='MLP hidden width')
    parser.add_argument('--layers', type=int, default=4, help='MLP hidden layers')
    parser.add_argument('--nsim', type=int, default=2000, help='samples per split')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--grad-inference', choices=['on', 'off', 'both'], default='both',
                        help='whether the dev loss carries an autograd graph')
    args = parser.parse_args()

    modes = {'on': [True], 'off': [False], 'both': [True, False]}[args.grad_inference]
    for grad_inference in modes:
        run(args.epochs, args.hsize, args.layers, args.nsim, args.batch, grad_inference)


if __name__ == '__main__':
    main()
