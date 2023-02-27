from neuromancer.psl.plot import *
import neuromancer.psl.autonomous as auto
import neuromancer.psl.nonautonomous as nauto
import neuromancer.psl.coupled_systems as cs
from neuromancer.psl.perturb import *

systems = {**auto.systems, **nauto.systems, **cs.systems}
emulators = systems

