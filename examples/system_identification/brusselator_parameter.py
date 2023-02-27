"""
Parameter estimation for a 1D Brusselator system.
"""
import torch
import neuromancer.slim as slim
import neuromancer.psl as psl
import numpy as np
import matplotlib.pyplot as plt
import argparse

from neuromancer import blocks, estimators, dynamics, arg, integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
import neuromancer.simulator as sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cpu"

    # %%  ground truth system
    system = psl.systems['Brusselator1D']
    ts = 0.05
    modelSystem = system()
    raw = modelSystem.simulate(ts=ts)
    psl.plot.pltOL(Y=raw['Y'])
    psl.plot.pltPhase(X=raw['Y'])

    # %%  Train, Development, Test sets - nstep and loop format
    nsteps = 1
    nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps, moving_horizon=False)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # %% fully observable estimator
    nx = 2
    estim = estimators.FullyObservable(
        {**train_data.dataset.dims, "x0": (nx,)},
        linear_map=slim.maps['identity'],
        input_keys=["Yp"])

    # %% Instantiate the ODE and symbolic dynamics model:
    brussels = ode.BrusselatorParam()
    fxRK4 = integrators.RK4(brussels, h=ts)
    fy = slim.maps['identity'](nx, nx)
    dynamics_model = dynamics.ODEAuto(fxRK4, fy, name='dynamics',
                            input_key_map={"x0": estim.output_keys[0]})

    # %% Constraints + losses:
    yhat = variable(dynamics_model.output_keys[2])
    y = variable("Yf")
    x0 = variable(estim.output_keys[0])
    xhat = variable(dynamics_model.output_keys[1])

    yFD = (y[:, 1:, :] - y[:, :-1, :])
    yhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])


    fd_loss = 2.0*((yFD == yhatFD)^2)
    fd_loss.name = 'FD_loss'

    reference_loss = ((yhat == y)^2)
    reference_loss.name = "ref_loss"

    # %%
    objectives = [reference_loss, fd_loss]
    constraints = []
    components = [estim, dynamics_model]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()
    problem = problem.to(device)

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout="nstep_dev_"+reference_loss.output_keys[0])

    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        patience=10,
        warmup=10,
        epochs=args.epochs,
        eval_metric="nstep_dev_"+reference_loss.output_keys[0],
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        logger=logger,
        device=device,
    )
    # %%
    best_model = trainer.train()
    # %%
    best_outputs = trainer.test(best_model)
    # os.system('cp test/open_loop.png figs/brusselator_parameter.png')
    # %%
    print('alpha = '+str(brussels.alpha.item()))
    print('beta = '+str(brussels.beta.item()))

    """
    Test open loop performance
    """
    nm_system = sim.DynamicsNeuromancer(dynamics_model,
                    name='nm', input_key_map={'x': 'x_nm'})
    psl_system = sim.DynamicsPSL(modelSystem,
                    name='psl', input_key_map={'x': 'x_psl'})
    psl_system.model.ts = ts
    components = [nm_system, psl_system]
    system = sim.SystemSimulator(components)
    x0 = np.ones(nx)
    data_init = {'x_psl': x0, 'x_nm': x0}
    trajectories = system.simulate(nsim=400, data_init=data_init)
    plt.close('all')
    psl.plot.pltOL(Y=trajectories['y_psl'], Ytrain=trajectories['y_nm'])
    plt.show(block=True)
    plt.interactive(False)

