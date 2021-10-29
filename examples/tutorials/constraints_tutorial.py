"""
Variable and Constraint tutorial

This script demonstrates how to create NeuroMANCER variables and constraints
that support declaration and evaluation of algebraic expressions as nn.Modules via operator overload
This high-level abstraction allows for intuitive formulation of constrained optimization problems

# Supported binary math operations which take Variable
# and numeric or Variable and Variable and return a Variable
# are +, -, *, @, **, also unary negation, -
# Supported comparison operations actiong on variables and return a Constraint are <, >, <=, >=, ==

"""

import neuromancer as nm
import torch
from neuromancer.constraint import Variable, Constraint, Objective, Loss

# Let's make a Variable to use in defining a constraint
x = Variable('x')
# Let's create a constraint by comparing variable and a constant
cnstr = x < 1.0
# Let's evaluate constraint violation at a given value of variable
print(cnstr({'x': 5.00}))
# take the value of the constraint violation
print(cnstr({'x': 5.00})[cnstr.name])

# Let's create a dataset dictionary with randomly sampled variable x
data = {'x': torch.rand([2,2,3])}
# and define new variable with initial value
a = Variable('a', value=1.5)
# now let's create new variable as algebraic expression of variables
math_exp_var = (3*x + 1 - 0.5 * a)**2
# now we create new constraint with 2-norm penalty on constraints violations
cnstr2 = (math_exp_var < 2.0)^2
# and evaluate its aggregate violations on dataset with random variable x
cnstr2(data)

# creating new variable y and new constraint with 2-norm penalty multiplied by torch parameter
y = Variable('y')
cnstr3 = torch.nn.Parameter(torch.tensor(0.1))*(x == y)^2
cnstr3({'x': torch.tensor(1.0), 'y': torch.tensor(0.5)})

# creating objective functions from variables and objective object
loss1 = math_exp_var.minimize(weight=10, name='loss1')
loss2 = Objective(math_exp_var, weight=10, name='loss2')
# creating objective via lambda functions through loss object
loss3 = Loss(
    ['a','x'],
    lambda a, x: torch.mean((3*x+1-0.5*a)**2),
    weight=10,
    name="loss3",
)