"""
Tutorial on constructing KKT conditions as Neuromancer constraints
"""

import neuromancer as nm
import torch
from neuromancer.constraint import Variable, Constraint, Objective
from neuromancer import policies
from neuromancer.gradients import gradient, jacobian, Gradient


"""
compute gradients of loss functions via nm.Objective class
"""
# Let's create a dataset dictionary with randomly sampled datapoints for parameter p
nsim = 20
data2 = {'p': torch.rand([nsim, 3], requires_grad=True)}
dims = {}
dims['p'] = data2['p'].shape
dims['U'] = (nsim, 2)  # defining expected dimensions of the solution variable: internal policy key 'U'
# create neural model
sol_map = policies.MLPPolicy(
    {**dims},
    hsizes=[10] * 2,
    input_keys=["p"],
    name='sol_map',
)
# make a forward pass on the component model
out = sol_map(data2)
# concatenate sampled dataset with model output
data3 = {**out, **data2}
# define variable z as output of the neural model
z = Variable(f"U_pred_{sol_map.name}", name='z')

# define loss on the component model outputs expression
loss = (z**2 + 5).minimize()
# define equuality constraint
con_eq = 3*(z**2 == 10)
# define inequality constraint
con_ineq = 3*(z >= 3)

# forward pass on the loss and constraints
l_out = loss(data3)
eq_out = con_eq(data3)
ineq_out = con_ineq(data3)
data4 = {**data3, **l_out}

# create variables as proxies to constraints and loss
l_var = Variable(loss.name, name='l_var')
eq_var = Variable(con_eq.name, name='eq_var')
ineq_var = Variable(con_ineq.name, name='ineq_var')

# compute gradient of the loss w.r.t. variable z via gradient function
dl_dz = gradient(loss(data3)[loss.name], data3[z.key])
print(dl_dz)
# compute gradient of the loss w.r.t. variable z grad method on nm.Objective
dl_dz_2 = loss.grad(data3, input_key=z.key)
print(dl_dz_2)

# create a new symbolic gradient variable dl/dz
dl_dz = l_var.grad(z)
# evaluate dl/dz on the dataset
dl_dz_val = dl_dz(data4)
print(dl_dz_val)

