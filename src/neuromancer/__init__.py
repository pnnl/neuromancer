import importlib.metadata

__version__ = importlib.metadata.version("neuromancer")

from neuromancer import dynamics
from neuromancer import modules

from neuromancer import system
from neuromancer import modules
from neuromancer import dataset
from neuromancer import constraint
from neuromancer import loss
from neuromancer import problem
from neuromancer import trainer
from neuromancer import plot

from neuromancer.dynamics import *
from neuromancer.modules import *

from neuromancer.system import *
from neuromancer.modules import *
from neuromancer.dataset import *
from neuromancer.constraint import *
from neuromancer.loss import *
from neuromancer.problem import *
from neuromancer.trainer import *
from neuromancer.plot import *