#%%
from neuromancer.interpolation import LinInterp_Offline
from matplotlib import pyplot as plt
import torch

# %% uniform sampling in t
t = (torch.arange(100)*0.1).unsqueeze(-1)
u = torch.sin(t)

#%%
interp = LinInterp_Offline(t, u)

tq = torch.tensor([[8.845]])
uq = interp(tq)

tq_vec = (torch.arange(5)*0.73 + 2).unsqueeze(-1)
uq_vec = interp(tq_vec)
# %%
t_plot = t.numpy()
u_plot = u.numpy()

tq_plot = tq.numpy()
uq_plot = uq.numpy()

tq_vec_plot = tq_vec.numpy()
uq_vec_plot = uq_vec.numpy()
# %%
plt.figure()
plt.scatter(t_plot, u_plot, s = 1, label = 'the ref')
plt.scatter(tq_plot, uq_plot, s = 8, c = 'k', label = 'the interp')
plt.scatter(tq_vec_plot, uq_vec_plot, s = 8, label = 'the interp (vectorized)')
plt.title('if t is uniformly sampled')
plt.xlabel('time')
plt.legend()
plt.show(block=True)
plt.interactive(False)
