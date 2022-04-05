import os
from neuromancer.integrators import integrators


systems = ["LorenzSystem", "VanDerPol", "ThomasAttractor", "RosslerAttractor", "Brusselator1D"]
ts = {"LorenzSystem": 0.01, "VanDerPol": 0.1,
      "ThomasAttractor": 0.1, "RosslerAttractor": 0.1,
      "Brusselator1D": 0.1}
steppers = [k for k in integrators]

for i in range(1):
    for system in systems:
        for stepper in steppers:
            os.system(f'python auto_timestepper.py -logdir {i}_{stepper}_{system} '
                      f'-exp autonomous -location mlruns -stepper {stepper} -system {system} '
                      f'-nsteps 4 -nsim 100 -batch_size 100')