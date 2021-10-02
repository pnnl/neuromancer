"""
Tutorial on constructing KKT conditions as Neuromancer constraints
"""

import neuromancer as nm
import torch
from neuromancer.constraint import Variable, Constraint, Objective
from neuromancer import policies
from neuromancer.gradients import gradient, jacobian, Gradient


"""
Define dataset, component model (primal solution map), constraints, and objective
"""
# Let's create a dataset dictionary with randomly sampled datapoints for parameter p
nsim = 20
data = {'p': torch.rand([nsim, 3], requires_grad=True)}
dims = {}
dims['p'] = data['p'].shape
dims['U'] = (nsim, 2)  # defining expected dimensions of the solution variable: internal policy key 'U'
# create primal solution map as neural model
primal_sol_map = policies.MLPPolicy(
    {**dims},
    hsizes=[10] * 2,
    input_keys=["p"],
    name='primal_sol_map',
)
# define variable z as output of the neural model
z = Variable(f"U_pred_{primal_sol_map.name}", name='z')

# define objective on the component model outputs expression
f = (z[:, :, [0]]**2 + z[:, :, [1]]**2 + 5)
obj = f.minimize()
# define equuality constraint
h = z[:, :, [0]] + z[:, :, [1]]**2 - 10
con_eq = 3*(h == 0)
# define inequality constraint
g = z[:, :, [0]]**2 + z[:, :, [1]]**2 - 3
con_ineq = 3*(g >= 0)

"""
Make forward pass on the constructed component, constraints, and objective functions
"""
# make a forward pass on the primal model
out = primal_sol_map(data)
# concatenate sampled dataset with model output
data1 = {**out, **data}
# forward pass on the loss and constraints
f_out = obj(data1)
eq_out = con_eq(data1)
ineq_out = con_ineq(data1)
# concatenate output dictionaries
data2 = {**data1, **f_out, **eq_out, **ineq_out}

print(f(data1).shape)
print(g(data1).shape)
print(h(data1).shape)

print(f_out[obj.name].shape)
print(eq_out[con_eq.name].shape)
print(ineq_out[con_ineq.name].shape)

"""
Create neuromancer derivatives to objective and constraints and their derivatives
"""

# create a new symbolic gradient variable dl/dz
df_dz = f.grad(z)
# evaluate dl/dz on the dataset
df_val = df_dz(data2)
print(df_val.shape)
# create a new symbolic gradient variable dh/dz
dh_dz = h.grad(z)
# evaluate dl/dz on the dataset
dh_val = dh_dz(data2)
print(dh_val.shape)
# create a new symbolic gradient variable dg/dz
dg_dz = g.grad(z)
# evaluate dl/dz on the dataset
dg_val = dg_dz(data2)
print(dg_val.shape)

# create variables as proxies to constraints and objective penalties
l_var = Variable(obj.name, name='l_var')
eq_var = Variable(con_eq.name, name='eq_var')
ineq_var = Variable(con_ineq.name, name='ineq_var')

# create a new symbolic gradient variable dobj/dz
dobj_dz = l_var.grad(z)
# evaluate dl/dz on the dataset
dobj_val = dobj_dz(data2)
print(dobj_val.shape)
# create a new symbolic gradient variable deq/dz
deq_dz = eq_var.grad(z)
# evaluate dl/dz on the dataset
deq_val = deq_dz(data2)
print(deq_val.shape)
# create a new symbolic gradient variable dineq/dz
dineq_dz = ineq_var.grad(z)
# evaluate dl/dz on the dataset
dineq_val = dineq_dz(data2)
print(dineq_val.shape)

"""
Define dual solution map and dual variables
"""
# create dual solution map as neural model
dims['U'] = (nsim, 2)   # 2 dual variables
dual_sol_map = policies.MLPPolicy(
    {**dims},
    hsizes=[10] * 2,
    input_keys=["p"],
    name='dual_sol_map',
)
# make a forward pass on the dual model
out2 = dual_sol_map(data)
data3 = {**data2, **out2}
# define dual variables (Lagrange multipliers): lambda and mu
lambd = Variable(f"U_pred_{dual_sol_map.name}", name='lambda')[:,:,[0]]
mu = Variable(f"U_pred_{dual_sol_map.name}", name='mu')[:,:,[1]]

"""
Formulate KKT conditions as neuromancer constraints 
https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions
"""

# VERSION 1: KKT conditions via loss and constraints functions gradients - computes gradients explicitly

# stationarity
stat = df_dz + mu*dg_dz + lambd*dh_dz == 0
# dual feasibility
dual_feas = mu >= 0
# complementarity slackness
comp_slack = mu*g == 0

# evaluate KKT conditions 1
stat_out = stat(data3)
dual_feas_out = dual_feas(data3)
comp_slack_out = comp_slack(data3)
print(stat_out[stat.name])
print(dual_feas_out[dual_feas.name])
print(comp_slack_out[comp_slack.name])

# evaluate shapes of KKT conditions constituents
print(f(data3).shape)
print(g(data3).shape)
print(h(data3).shape)
print(df_dz(data3).shape)
print(dg_dz(data3).shape)
print(dh_dz(data3).shape)
print(lambd(data3).shape)
print(mu(data3).shape)

# VERSION 2: KKT conditions with aggregated loss and constraints penalties - computes gradients implicitly

# stationarity
stat2 = dobj_dz + mu*deq_dz + lambd*dineq_dz == 0
# dual feasibility
dual_feas2 = mu >= 0
# complementarity slackness
comp_slack2 = mu*g == 0

# evaluate KKT conditions 2
stat_out2 = stat2(data3)
dual_feas_out2 = dual_feas2(data3)
comp_slack_out2 = comp_slack2(data3)
print(stat_out2[stat2.name])
print(dual_feas_out2[dual_feas2.name])
print(comp_slack_out2[comp_slack2.name])

# evaluate shapes of KKT conditions constituents
print(dobj_dz(data3).shape)
print(deq_dz(data3).shape)
print(dineq_dz(data3).shape)
print(ineq_var(data3).shape)
print(lambd(data3).shape)
print(mu(data3).shape)

