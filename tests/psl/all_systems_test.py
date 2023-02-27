from neuromancer.psl import plot
import matplotlib.pyplot as plt
import os

import neuromancer.psl.autonomous as auto
import neuromancer.psl.nonautonomous as nauto
from neuromancer.psl.coupled_systems import Boids, Gravitational_System, RC_Network
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    """
    Tests
    """
    os.system('rm -rf figs')
    os.mkdir("./figs")
    ninit = 0

    for name, system in nauto.systems.items():
        print(name)
        model = system(nsim=1000)
        out = model.simulate()
        plot.pltOL(Y=out['Y'], U=out['U'], figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'], figname="./figs/"+name+"_phase")
        plt.close('all')

    for name, system in auto.systems.items():
        print(name)
        model = system()
        out = model.simulate()
        plot.pltOL(Y=out['Y'], figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'], figname="./figs/"+name+"_phase")
        plt.close('all')

    for name, system in [('RC_Network', RC_Network)]:
        print(name)
        model = system(nsim=1000)
        out = model.simulate()
        plot.pltOL(Y=out['Y'], U=out['U'], figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'], figname="./figs/"+name+"_phase")
        plt.close('all')

    for name, system in [('Gravitational_system', Gravitational_System), ('Boids', Boids)]:
        print(name)
        model = system(nsim=100)
        out = model.simulate(nsim=100)
        plot.pltOL(Y=out['Y'].reshape(101, -1), figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'].reshape(101, -1), figname="./figs/"+name+"_phase")
        plt.close('all')

