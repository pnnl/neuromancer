import torch

from neuromancer.constraint import variable
from neuromancer.loss import AugmentedLagrangeLoss

weight = 5

datapoints = {"p": torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]),
              "x": torch.tensor([[0., 0.], [0., 0.], [0., 0.]]),
              "y": torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]),
              "name": "test"}

# parameters
p = variable("p")
p1, p2 = p[:, 0], p[:, 1]

# problem 1:
# variables
x = variable("x")
x1, x2 = x[:, 0], x[:, 1]
# objective function
f = x1 ** 2 + x2 ** 2
obj = [f.minimize(weight=1.0, name="obj")]
# constraints
Q_con = weight
con_1 = Q_con * (x1 + x2 - p1 >= 0)
con_1.name = "c1"
con_2 = Q_con * (x1 + x2 - p1 <= 5)
con_2.name = "c2"
con_3 = Q_con * (x1 - x2 + p2 <= 5)
con_3.name = "c3"
con_4 = Q_con * (x1 - x2 + p2 >= 0)
con_4.name = "c4"
cons = [con_1, con_2, con_3, con_4]
# barrier loss
loss_1 = AugmentedLagrangeLoss(obj, cons, datapoints)

# problem 2
y = variable("y")
y1, y2 = y[:, 0], y[:, 1]
# objective function
f = (1 - y1) ** 2 + p1 * (y2 - y1 ** 2) ** 2
obj = [f.minimize(weight=1.0, name="obj")]
# constraints
Q_con = weight
con_1 = Q_con * (y1 >= y2)
con_1.name = "c1"
con_2 = Q_con * (y1 ** 2 + y2 ** 2 >= p2 / 2)
con_2.name = "c2"
con_3 = Q_con * (y1 ** 2 + y2 ** 2 <= p2)
con_3.name = "c3"
cons = [con_1, con_2, con_3]
# penalty loss
loss_2 = AugmentedLagrangeLoss(obj, cons, datapoints)

# aggregate loss
loss = 0.1 * loss_1 + 1.0 * loss_2

# test
datapoints = {"p": torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]),
              "x": torch.tensor([[0., 0.], [0., 0.], [0., 0.]]),
              "y": torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]),
              "name": "test"}
print(loss(datapoints)["loss"])