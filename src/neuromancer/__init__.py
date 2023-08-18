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
