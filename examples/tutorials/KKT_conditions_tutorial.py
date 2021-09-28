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
obj = (z**2 + 5).minimize()
# define equuality constraint
con_eq = 3*(z**2 == 10)
# define inequality constraint
con_ineq = 3*(z >= 3)

"""
Make forward pass on the constructed component, constraints, and loss
"""
# make a forward pass on the primal model
out = primal_sol_map(data)
# concatenate sampled dataset with model output
data1 = {**out, **data}
# forward pass on the loss and constraints
l_out = obj(data1)
eq_out = con_eq(data1)
ineq_out = con_ineq(data1)
# concatenate output dictionaries
data2 = {**data1, **l_out, **eq_out, **ineq_out}

"""
Create neuromancer variables as proxies to objective and constraints and their derivatives
"""
# create variables as proxies to constraints and objective
l_var = Variable(obj.name, name='l_var')
eq_var = Variable(con_eq.name, name='eq_var')
ineq_var = Variable(con_ineq.name, name='ineq_var')

# create a new symbolic gradient variable dl/dz
dl_dz = l_var.grad(z)
# evaluate dl/dz on the dataset
dl_dz_val = dl_dz(data2)
print(dl_dz_val)

# create a new symbolic gradient variable deq_var/dz
deq_dz = eq_var.grad(z)
# evaluate dl/dz on the dataset
deq_var_val = deq_dz(data2)
print(deq_var_val)

# create a new symbolic gradient variable dineq_var/dz
dineq_dz = ineq_var.grad(z)
# evaluate dl/dz on the dataset
dineq_var_val = dineq_dz(data2)
print(dineq_var_val)

"""
Define dual solution map and dual variables
"""
# create dual solution map as neural model
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
# stationarity
stat = dl_dz + mu*dineq_dz + lambd*deq_dz == 0
# dual feasibility
dual_feas = mu >= 0
# complementarity slackness
g = con_ineq.weight*(con_ineq.left - con_ineq.right)      # g <= 0
comp_slack = mu*g == 0

# evaluate KKT conditions
stat_out = stat(data3)
dual_feas_out = dual_feas(data3)
comp_slack_out = comp_slack(data3)
print(stat_out[stat.name])
print(dual_feas_out[dual_feas.name])
print(comp_slack_out[comp_slack.name])

# TODO check if the numerics holds
# kl = mu*dineq_dz
# mu(data3)
# lambd(data3)
# dineq_dz(data3)
# g(data3)