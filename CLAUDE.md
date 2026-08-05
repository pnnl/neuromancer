# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NeuroMANCER is a PyTorch library for differentiable programming over constrained optimization,
system identification, and control. The package source is under `src/neuromancer/` and installs as
the `neuromancer` package.

## Commands

Install for development (from repo root):

```bash
pip install -e .[docs,tests,examples]   # zsh: pip install -e .'[docs,tests,examples]'
```

Run the test suite:

```bash
pytest tests/                       # ~2 minutes on CPU, emits many warnings
pytest tests/test_problem.py        # single file
pytest tests/test_problem.py::test_name -x
```

`tests/run_examples.py` is not part of the default suite in practice: it walks `examples/`
recursively and runs every `.py` file with `-epochs 1`, appending output to `results.txt`.

Build docs (Sphinx, also run by `.github/workflows/update_docs.yml` on push to `master`):

```bash
sphinx-build docs html
```

## Architecture

The library composes four layers. Understanding the dictionary-passing contract between them
explains most of the code.

**1. Dictionary-passing computation.** Every computational unit takes `Dict[str, Tensor]` and
returns `Dict[str, Tensor]`. `Node` (`system.py`) wraps an arbitrary callable, gathering its
positional arguments from `input_keys` and writing its returns to `output_keys`. This is the only
adapter needed to put a plain `nn.Module` into a NeuroMANCER graph.

**2. Symbolic variables and constraints.** `Variable` (`constraint.py`) is a lazily evaluated node
in a networkx DAG. `variable('x')` creates an input variable that retrieves `data['x']`; arithmetic
and indexing on variables build new variables; comparison operators (`<=`, `>=`, `==`) return
`Constraint` objects; `.minimize()` returns an `Objective`. Constraint classes `LT`, `GT`, `Eq`
compute a triple `(loss, value, penalty)` where penalty is the ReLU'd violation. Multiplying a
constraint by a scalar sets its weight. `variable` is a `plum` multiple-dispatch function with
overloads for key strings, sizes, tensors, and `(inputs, func)` pairs.

**3. Loss aggregation.** `AggregateLoss` subclasses in `loss.py` (`PenaltyLoss`, `BarrierLoss`,
`AugmentedLagrangeLoss`) take lists of objectives and constraints and produce the keys `loss`,
`objective_loss`, `penalty_loss`, plus constraint value/violation summaries.

**4. Problem and rollout.** `Problem` (`problem.py`) runs a list of nodes in sequence over a merging
dictionary, then applies the loss. `System` (`system.py`) runs nodes in a loop over `nsteps`,
concatenating each step's 2-D `(batch, dim)` node outputs into 3-D `(batch, time, dim)` tensors —
this is how cyclic (closed-loop) graphs are expressed. `System` slices `data[k][:, i]` per step, so
node callables see 2-D tensors while dataset tensors are 3-D. `SystemPreview` extends this with a
future window of known variables.

### Key naming contract

`Problem.forward` prefixes every output key with `data["name"]`, the name carried on the
`DictDataset`. A dataset named `train` therefore produces `train_loss`, and `dev` produces
`dev_loss`. Trainers select metrics by those exact prefixed names (`train_metric='train_loss'`,
`dev_metric='dev_loss'`). `LitDataModule.setup` asserts the train and dev datasets are named
`train` and `dev`.

Node, objective, and constraint names must be unique within a `Problem` or `System`; both call
`_check_unique_names` when constructing their pydot graph in `__init__`.

### Supporting modules

- `modules/blocks.py` — neural architectures (`MLP`, `ResMLP`, `KANBlock`, `InputConvexNN`,
  `PosDef`, `RNN`, `Transformer`), all implementing the `Block` interface.
- `slim/` — structured linear maps as drop-in `nn.Linear` replacements, selected through
  `slim.maps['name']` (`linear`, `l0`, spectral, symplectic, Perron-Frobenius, SVD, and others).
  Passed to blocks as `linear_map=`.
- `dynamics/` — `ODESystem` and `BaseSDESystem` bases for grey-box and physics-informed models,
  `integrators.py` (fixed-step RK variants, symplectic integrators, `DiffEqIntegrator` over
  torchdiffeq, SDE integrators over torchsde), `library.py` for SINDy function libraries.
- `psl/` — physics simulation library of reference ODE/PDE systems for data generation and
  benchmarking. `psl.systems` is the registry merging autonomous, nonautonomous, building envelope,
  and coupled systems.
- `dataset.py` — `DictDataset` is the standard container; `SequenceDataset`, `StaticDataset`,
  `GraphDataset` and the `get_*_dataloaders` helpers handle normalization and splitting.

### Two training paths

`trainer.py` has both. `Trainer` is the plain PyTorch loop taking `Problem` plus DataLoaders, with
`Callback` hooks at batch/epoch/eval boundaries; it returns the best state dict and keeps the
best weights on the model. `LitTrainer` subclasses `pl.Trainer` and takes a `Problem` plus a
`data_setup_function` that returns `(train_data, dev_data, test_data, batch_size)`; it wraps them
in `LitProblem` and `LitDataModule`. The USER_GUIDE directs new contributions toward `LitTrainer`.

## Conventions

- Tests are lightweight pytest functions in `tests/`, with `tests/psl/` and `tests/slim/` for those
  subpackages. Contributions are expected to come with tests plus a runnable script or notebook in
  `examples/`.
- New methods or domain applications are expected to include a tutorial notebook in the matching
  `examples/` subfolder (`control`, `ODEs`, `PDEs`, `SDEs`, `DAEs`, `KANs`, `parametric_programming`,
  `domain_examples`, `tutorials`, `lightning_integration_examples`, `function_encoder`).
- Example scripts accept `-epochs` and other flags through `arg.py`, which `tests/run_examples.py`
  relies on.
- Graph plotting (`Problem.show()`, `System.show()`) needs Graphviz installed system-wide.
- `mlflow` and `wandb` are optional extras under `tracking`, not core dependencies.
