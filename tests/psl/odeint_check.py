from neuromancer.psl.autonomous import systems
from scipy.integrate import odeint as sodeint
from torchdiffeq import odeint as todeint
import numpy as np
import torch

for sys in systems.values():
    print(sys)
    system = sys(backend='torch')
    print(system.ts, system.nsim, system.ninit)
    data = system.simulate(nsim=10)
    print(data['X'].shape, data['Y'].shape)
    print(data['X'])