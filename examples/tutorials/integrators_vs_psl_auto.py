#%%
import neuromancer.psl as psl
from neuromancer import integrators
from neuromancer import ode
import torch
import numpy as np
import matplotlib.pyplot as plt

#%% autonomous system
# PSL simulation
system = psl.systems['Brusselator1D']

modelSystem = system()
ts = 0.001
nsim = 10000
raw = modelSystem.simulate(ts=ts, nsim=nsim)
X = raw['X'][:-1, :]
Y = raw['Y'][:-1, :]
psl.plot.pltOL(Y=Y)
psl.plot.pltPhase(X=Y)

traj_ref = X

#%% single-step integrator simulation
ode_block = ode.BrusselatorParam()
ode_block.alpha = torch.nn.Parameter(torch.tensor([1.0]),
 requires_grad=False)

ode_block.beta = torch.nn.Parameter(torch.tensor([3.0]),
 requires_grad=False)

integrator = integrators.RK4(ode_block, interp_u=None, h=ts)

ic = torch.tensor(modelSystem.x0, dtype=torch.float32,
 requires_grad=False).reshape((1, X.shape[1]))
traj_list = [ic]

for step in range(nsim):
    traj_list.append(integrator(traj_list[step]))

traj_pred = torch.cat(traj_list[1:], dim=0).numpy()  # remove IC

err = np.linalg.norm(traj_pred - traj_ref)/np.linalg.norm(traj_ref)

print('single-step auto err')
print(err)

plt.figure()
plt.plot(traj_pred, 'o', label='pred')
plt.plot(traj_ref, label='ref')
plt.legend()

#%% multi-step integrator simulation
ic_multistep = torch.from_numpy(traj_ref[:4, :]).reshape((4, 1,
 X.shape[1]))

integrator_multistep = integrators.MultiStep_PredictorCorrector(ode_block,
 interp_u=None, h=ts)

input_list = [ic_multistep]
output_list = []
for step in range(nsim):
    output_list.append(integrator_multistep(input_list[step]))
    new_input = torch.cat([input_list[step][1:, :, :],
     output_list[step].unsqueeze(1)], dim=0)
    input_list.append(new_input)

traj_pred = torch.cat([ic_multistep.squeeze(1)] + output_list,
 dim = 0).squeeze(1).numpy() 
err = np.linalg.norm(traj_pred[:-4, :] - traj_ref)/np.linalg.norm(traj_ref)

print('multi-step auto err')
print(err)

plt.figure()
plt.plot(traj_pred[:-4, :], 'o', label='pred')
plt.plot(traj_ref, label='ref')
plt.legend()

#%%