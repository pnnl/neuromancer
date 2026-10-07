"""One registry of surfaceable factories, by category.

Aggregates the per-module registries — ``blocks.blocks``,
``integrators.integrators`` with the multistep and second-order sets,
``ode.odes``, ``slim.maps``, and ``activations.activations`` — into one
dictionary for tools that enumerate what neuromancer can construct. Keys
within a category are the names the per-module registries use; every value is
the class itself.
"""

from neuromancer.dynamics import integrators, ode
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.slim.linear import maps

registry = {
    "blocks": dict(blocks.blocks),
    "integrators": {
        **integrators.integrators,
        **integrators.integrators_multistep,
        **integrators.integrators_second_order,
    },
    "dynamics": dict(ode.odes),
    "slim": dict(maps),
    "activations": dict(activations),
}
