import numpy as np
import torch

from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss

from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


# parameterized quadratic programming
def lossQuadratic(aggLoss, weight):
    # parameters
    p = variable("p")
    p1, p2 = p[:, 0], p[:, 1]
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
    # loss
    loss = aggLoss(obj, cons)
    return loss


# parameterized Rosenbrock problem
def lossRosenbrock(aggLoss, weight):
    # parameters
    p = variable("p")
    p1, p2 = p[:, 0], p[:, 1]
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
    # loss
    loss = aggLoss(obj, cons)
    return loss


agg_losses = [PenaltyLoss, BarrierLoss]
problems = [lossQuadratic, lossRosenbrock]


@given(arrays(np.float, (3,2), elements=st.floats(0.5, 1.2)),
       arrays(np.float, (3,2), elements=st.floats(0.0, 3.0)),
       arrays(np.float, (3,2), elements=st.floats(0.0, 1.0)),
       st.sampled_from(agg_losses),
       st.integers(0, 200))
@settings(max_examples=200, deadline=None)
def test_add(p, x, y, aggLoss, weight):
    # data points
    datapoints = {"p": torch.from_numpy(p),
                  "x": torch.from_numpy(x),
                  "y": torch.from_numpy(y),
                  "name": "test"}
    # loss for quadratic
    loss1 = lossQuadratic(aggLoss, weight)
    output1 = loss1(datapoints)
    # loss for Rosenbrock
    loss2 = lossRosenbrock(aggLoss, weight)
    output2 = loss2(datapoints)
    # add
    loss = loss1 + loss2
    output = loss(datapoints)
    # test
    assert torch.isclose(output1["objective_loss"] + output2["objective_loss"], output["objective_loss"])
    assert torch.isclose(output1["penalty_loss"] + output2["penalty_loss"], output["penalty_loss"])
    assert torch.isclose(output1["loss"] + output2["loss"], output["loss"])


@given(st.floats(0.1, 10),
       arrays(np.float, (3,2), elements=st.floats(0.5, 1.2)),
       arrays(np.float, (3,2), elements=st.floats(0.0, 3.0)),
       arrays(np.float, (3,2), elements=st.floats(0.0, 1.0)),
       st.sampled_from(agg_losses),
       st.sampled_from(problems),
       st.integers(1, 200))
@settings(max_examples=200, deadline=None)
def test_mul(multiplier, p, x, y, aggLoss, problems, weight):
    # data points
    datapoints = {"p": torch.from_numpy(p),
                  "x": torch.from_numpy(x),
                  "y": torch.from_numpy(y),
                  "name": "test"}
    # loss
    loss = problems(aggLoss, weight)
    output = loss(datapoints)
    # mul
    weighted_loss = multiplier * loss
    weighted_output = weighted_loss(datapoints)
    # test
    assert torch.isclose(multiplier * output["objective_loss"], weighted_output["objective_loss"])
    assert torch.isclose(multiplier * output["penalty_loss"], weighted_output["penalty_loss"])
    assert torch.isclose(multiplier * output["loss"], weighted_output["loss"])

