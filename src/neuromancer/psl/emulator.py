"""
Base Classes for dynamic systems emulators

"""
from abc import ABC, abstractmethod
import random
import numpy as np


class EmulatorBase(ABC):
    """
    base class of the emulator
    """
    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.nsim, self.ninit, self.ts = nsim, ninit, ts
        self.x0 = 0.

    @abstractmethod
    def equations(self, **kwargs):
        """
        Define equations defining the dynamical system
        """
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        """
        N-step forward simulation of the dynamical system
        """
        pass

