from data import BuildingDAE, disturbance, control_profile_DAE
from ssm import SSMGroundTruth
import numpy as np
import torch
from plot import plot_trajectories

M_flow, DT = control_profile_DAE(samples_day=288, sim_days=28)
#    manual turnoffs
M_flow[:, 3] = 0
M_flow[:, 4] = 0
M_flow[:, 5] = 0
nsim = M_flow.shape[0]
D = disturbance(n_sim=nsim)
building = BuildingDAE(rom=True)
X, Y = building.loop(nsim, M_flow, DT, D)
nx, nu, nd, ny, n_m, n_dT = building.nx, building.nu, building.nd, building.ny, building.nu, 1

model = SSMGroundTruth(nx, ny, n_m, n_dT, nu, nd, 1)
x0 = torch.tensor(0 * np.ones(building.nx, dtype=np.float32)) # initial conditions
X_pred, Y_pred, U_pred, regularization_error = model(x0, torch.tensor(M_flow, dtype=torch.float32),
                                                     torch.tensor(DT, dtype=torch.float32),
                                                     torch.tensor(D, dtype=torch.float32))
print(X_pred.shape, X.shape)
plot_trajectories([X[1:2001, 0], X[1:2001, 1]],
                  [X_pred[:2000, 0, 0].detach().cpu().numpy(), X_pred[:2000, 0, 1].detach().cpu().numpy()],
                  ['x0', 'x1'],
                  'test1.png')
print(np.mean(X[1:2001, 0] - X_pred[:2000, 0, 0].detach().cpu().numpy()))

print(Y_pred.shape, Y.shape)
plot_trajectories([Y[1:2001, 0], Y[1:2001, 1]],
                  [Y_pred[:2000, 0, 0].detach().cpu().numpy(), Y_pred[:2000, 0, 1].detach().cpu().numpy()],
                  ['x0', 'x1'],
                  'test2.png')
print(np.mean(Y[1:2001, 0] - Y_pred[:2000, 0, 0].detach().cpu().numpy()))




