from neuromancer.psl import plot, systems

if __name__ == '__main__':
    """
    Test and plot trajectories of a single system
    """
    system = systems['Duffing']
    model = system()
    out = model.simulate()

    Y = out['Y']
    X = out['X']
    U = out['U'] if 'U' in out.keys() else None
    D = out['D'] if 'D' in out.keys() else None
    plot.pltOL(Y=Y, X=X, U=U, D=D)
    plot.pltPhase(X=Y)
