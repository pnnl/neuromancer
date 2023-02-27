"""
Modeling a Two Tank system with neural ODEs
"""

import torch
import numpy as np
import neuromancer.slim as slim
import neuromancer.psl as psl
import matplotlib.pyplot as plt

from neuromancer import blocks, estimators, dynamics, arg, integrators, ode
from neuromancer.activations import activations
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
import neuromancer.simulator as sim

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cpu"
    system = psl.systems['TwoTank']
    ts = 4.0
    nsim = 10000
    modelSystem = system(ts=ts, nsim=nsim)
    modelSystem.ts = ts
    raw = modelSystem.simulate(ts=ts)
    raw['Y'] = raw['Y'][:-1, :]
    raw['X'] = raw['X'][:-1, :]
    # psl.plot.pltOL(Y=raw['Y'], U=raw['U'])
    # psl.plot.pltPhase(X=raw['Y'])

    # #%% Interpolant for offline mode
    interp_u = lambda tq, t, u: u

    #%%
    #  Train, Development, Test sets - nstep and loop format
    nsteps = 60  # nsteps rollouts in training
    nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps,
                                moving_horizon=True)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # %% Identity mapping
    nx = dims['X'][1]
    estim = estimators.FullyObservable(
        {**train_data.dataset.dims, "x0": (nx,)},
        linear_map=slim.maps['identity'],
        input_keys=["Yp"],
    )
    estim(train_data.dataset.get_full_batch())

    # %% Instantiate the blocks, dynamics model:
    nu = dims['U'][1]
    black_box_ode = blocks.MLP(insize=nx+nu, outsize=nx, hsizes=[32, 32],
                               linear_map=slim.maps['linear'],
                               nonlin=activations['relu'])
    fx_int = integrators.RK4(black_box_ode, interp_u=interp_u, h=modelSystem.ts)
    fy = slim.maps['identity'](nx, nx)

    dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                    input_key_map={"x0": estim.output_keys[0],
                                   'Yf': 'Yf', 'Uf': 'Uf'},
                    name='dynamics', online_flag=False)

    # %% Constraints + losses:
    yhat = variable(dynamics_model.output_keys[2])
    y = variable("Yf")
    yFD = (y[:, 1:, :] - y[:, :-1, :])
    yhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])
    fd_loss = (yFD == yhatFD)^2
    fd_loss.name = 'FD_loss'
    reference_loss = 10*((yhat == y)^2)
    reference_loss.name = "ref_loss"
    Q_con = 10.
    state_lower_bound_penalty = Q_con * (yhat > -0.05)
    state_upper_bound_penalty = Q_con * (yhat < 1.2)
    state_lower_bound_penalty.name = 'y_min'
    state_upper_bound_penalty.name = 'y_max'
    # %%
    objectives = [reference_loss, fd_loss]
    constraints = [state_lower_bound_penalty,
                   state_upper_bound_penalty]
    components = [estim, dynamics_model]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()
    problem = problem.to(device)

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        patience=100,
        warmup=10,
        epochs=200,
        eval_metric="nstep_dev_"+reference_loss.output_keys[0],
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        device=device)
    # %%
    best_model = trainer.train()
    # %%
    best_outputs = trainer.test(best_model)
    # %%

    """
    Test open loop performance
    """
    nm_system = sim.DynamicsNeuromancer(dynamics_model,
                    name='nm', input_key_map={'x': 'x_nm', 'u': 'U'})
    psl_system = sim.DynamicsPSL(modelSystem,
                    name='psl', input_key_map={'x': 'x_psl', 'u': 'U'})
    psl_system.model.ts = ts
    components = [nm_system, psl_system]
    system_sim = sim.SystemSimulator(components)
    sim_steps = 2000
    modelSystem = system(ts=ts, nsim=sim_steps+1)
    raw = modelSystem.simulate()
    x0 = np.asarray(modelSystem.x0)
    data_init = {'x_psl': x0, 'x_nm': x0}
    U = raw['U'][:sim_steps+1, :]
    data_traj = {'U': U}
    trajectories = system_sim.simulate(nsim=sim_steps, data_init=data_init,
                                       data_traj=data_traj)
    psl.plot.pltOL(Y=trajectories['y_psl'],
                   Ytrain=trajectories['y_nm'], U=trajectories['U'])
    plt.show(block=False)
    plt.interactive(False)
    plt.savefig('openloop.png')