"""
Open and closed loop dynamical models
x: states (x0 - initial conditions)
y: predicted outputs
ym: measured outputs
u: control inputs
d: uncontrolled inputs (measured disturbances)
r: reference signals

generic closed-loop dynamics:
x+ = fx(x) o fu(u) o fd(d)
y =  fy(x)
x = estim(ym,x0,u,d)
u = policy(x,u,d,r)

Dynamical models from ssm.py
estimator from estimators.py
policy from  policies.py
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import linear
import estimators
import policies
import ssm


# single forward time step of the open/closed loop dynamics
def step(loop, data):
    pass

class OpenLoop(nn.Module):
    def __init__(self, model=ssm.BlockSSM, stim=estimators.LinearEstimator,
                 Linear=linear.Linear, **linargs):
        """

        :param model: SSM mappings, see ssm.py
        :param estim: state estimator mapping, see estimators.py
        """
        super().__init__()



class ClosedLoop(nn.Module):
    def __init__(self, model=ssm.BlockSSM, estim=estimators.LinearEstimator,
                 policy=policies.LinearPolicy, Linear=linear.Linear, **linargs):
        """

        :param model: SSM mappings, see ssm.py
        :param estim: state estimator mapping, see estimators.py
        :param policy: policy mapping, see policies.py
        """
        super().__init__()


if __name__ == '__main__':
    pass