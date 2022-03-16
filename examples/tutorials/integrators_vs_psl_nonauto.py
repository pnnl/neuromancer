#%%
import psl
from neuromancer import integrators
from neuromancer import ode
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuromancer.interpolation import LinInterp_Offline, LinInterp_Online

#%% Nonautonomous system (online mode & offline mode)
# PSL simulation
system = psl.systems['Duffing']

modelSystem = system()
ts = 0.001
nsim = 5000
raw = modelSystem.simulate(ts=ts, nsim = nsim)
psl.plot.pltOL(Y=raw['Y'])
psl.plot.pltPhase(X=raw['Y'])

traj_ref = raw['X']
t_ref = (np.arange(nsim)*ts).reshape(-1, 1)

#%% single-step integrator simulation. offline.
ode_block = ode.DuffingParam()
ode_block.omega = torch.nn.Parameter(torch.tensor([0.5]),
 requires_grad=False)

t = torch.from_numpy(t_ref)
interp_u_offline = LinInterp_Offline(t, t)

integrator = integrators.RK4(ode_block, interp_u=interp_u_offline, h=ts)

ic = torch.tensor(modelSystem.x0, dtype=torch.float32,
 requires_grad=False).reshape((1, raw['X'].shape[1]))
traj_list = [ic]

for step in range(nsim):
    traj_list.append(integrator(traj_list[step], t[[step], :], t[[step], :]))

traj_pred = torch.cat(traj_list[1:], dim=0).numpy()  # remove IC

err = np.linalg.norm(traj_pred - traj_ref)/np.linalg.norm(traj_ref)

print('single-step nonauto offline err')
print(err)

plt.figure()
plt.plot(traj_pred, 'o', label='pred')
plt.plot(traj_ref, label='ref')
plt.legend()

#%% single-step integrator simulation. online.
interp_u_online = LinInterp_Online()
integrator = integrators.RK4(ode_block, interp_u=interp_u_online, h=ts)

ic = torch.tensor(modelSystem.x0, dtype=torch.float32,
 requires_grad=False).reshape((1, raw['X'].shape[1]))
traj_list = [ic]

for step in range(nsim-1):  # cannot do online interp for the last time step
    traj_list.append(integrator(traj_list[step], t[step:step+2, :].unsqueeze(1),
                                t[step:step+2, :].unsqueeze(1)))

traj_pred = torch.cat(traj_list[1:], dim=0).numpy()  # remove IC

err = np.linalg.norm(traj_pred - traj_ref[:-1, :])/np.linalg.norm(traj_ref[:-1, :])

print('single-step nonauto online err')
print(err)

plt.figure()
plt.plot(traj_pred, 'o', label='pred')
plt.plot(traj_ref, label='ref')
plt.legend()

#%% multi-step integrator simulation. offline.
integrator_multistep = integrators.MultiStep_PredictorCorrector(ode_block, interp_u=interp_u_offline, h=ts)
ic_multistep = torch.from_numpy(traj_ref[:4, :]).reshape((4, 1,
 raw['X'].shape[1]))

input_list = [ic_multistep]
output_list = []
for step in range(nsim-4):
    output_list.append(integrator_multistep(input_list[step], t[[4+step], :], t[[4+step], :]))
    new_input = torch.cat([input_list[step][1:, :, :],
     output_list[step].unsqueeze(1)], dim=0)
    input_list.append(new_input)

traj_pred = torch.cat([ic_multistep.squeeze(1)] + output_list,
 dim = 0).squeeze(1).numpy() 
err = np.linalg.norm(traj_pred - traj_ref)/np.linalg.norm(traj_ref)

print('multi-step nonauto offline err')
print(err)

plt.figure()
plt.plot(traj_pred, 'o', label='pred')
plt.plot(traj_ref, label='ref')
plt.legend()

#%%