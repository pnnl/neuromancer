import os
from neuromancer.integrators import integrators


steppers = [k for k in integrators]

for i in range(5):
    for stepper in steppers:
        os.system(f'python lorenz_nonauto_timestepper.py -logdir {i}_{stepper}_nonauto '
                  f'-exp nonautonomous -location mlruns -stepper {stepper} '
                  f'-nsteps 2 -nsim 1000 -batch_size 100')