from neuromancer.modules import activations
from neuromancer.modules import blocks
from neuromancer.modules import functions
# operators is not imported here: it imports neuraloperator, whose models
# import torch-harmonics at module load, and torch-harmonics ships no wheel
# for every platform. `import neuromancer.modules.operators` loads it on demand.
from neuromancer.modules import solvers
from neuromancer.modules import lopo
from neuromancer.modules import function_encoder
