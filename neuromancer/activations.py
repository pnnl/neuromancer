# machine learning/data science imports
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_exp(alpha, x):
    if alpha == 0.0:
        return x
    elif alpha < 0.0:
        return -torch.log(1 - alpha * (x + alpha)) / alpha
    else:
        return (torch.exp(alpha * x) - 1) / alpha + alpha


class SoftExponential(nn.Module):
    """
    Soft exponential activation: https://arxiv.org/pdf/1602.01321.pdf
    """

    def __init__(self, alpha=0.0, tune_alpha=True):
        """

        :param alpha:
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=tune_alpha)

    def forward(self, x):
        return soft_exp(self.alpha, x)


class BLU(nn.Module):
    """
    Bendable Linear Units: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8913972
    """
    def __init__(self, tune_alpha=False, tune_beta=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.epsilon = 1e-7 if tune_alpha else 0.0
        self.epsilon = nn.Parameter(torch.tensor(self.epsilon), requires_grad=tune_beta)

    def forward(self, x):
        return self.beta * (torch.sqrt(x * x + self.alpha * self.alpha + 1e-7) - self.alpha) + x


class APLU(nn.Module):
    """
    Adaptive Piecewise Linear Units: https://arxiv.org/pdf/1412.6830.pdf
    """
    def __init__(self, nsegments=2, alpha_reg_weight=1e-3,
                 beta_reg_weight=1e-3, tune_alpha=True, tune_beta=True):
        super().__init__()
        self.nsegments = nsegments
        self.alpha_reg_weight = alpha_reg_weight
        self.beta_reg_weight = beta_reg_weight
        self.alpha = nn.Parameter(torch.rand(nsegments), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.rand(nsegments), requires_grad=tune_beta)

    def reg_error(self):
        return self.alpha_reg_weight*torch.norm(self.alpha) + self.beta_reg_weight*torch.norm(self.beta)

    def forward(self, x):
        y = F.relu(x)
        for i in range(self.nsegments):
            y += self.alpha[i] * F.relu(-x + self.beta[i])
        return y


class PReLU(nn.Module):
    """
    Parametric ReLU: https://arxiv.org/pdf/1502.01852.pdf
    """
    def __init__(self, tune_alpha=True, tune_beta=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=tune_beta)

    def forward(self, x):
        neg = self.alpha * -F.relu(-x)
        pos = self.beta * F.relu(x)
        return neg + pos


class PELU(nn.Module):
    """
    Parametric Exponential Linear Units: https://arxiv.org/pdf/1605.09332.pdf
    """
    def __init__(self, tune_alpha=True, tune_beta=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=tune_beta)

    def forward(self, x):
        posx = F.relu(x)
        negx = F.relu(-x)
        return (self.alpha/self.beta)*posx + self.alpha*(torch.exp(negx/self.beta) - 1)


class RectifiedSoftExp(nn.Module):
    """
    Mysterious unexplained implementation of Soft Exponential ported from author's Keras code:
    https://github.com/thelukester92/2019-blu/blob/master/python/activations/softexp.py
    """
    def __init__(self, tune_alpha=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=tune_alpha)
        self.epsilon = 1e-7

    def forward(self, x):
        neg_alpha = F.relu(-torch.clamp(self.alpha, -1, 1)) + self.epsilon()
        pos_alpha = F.relu(torch.clamp(self.alpha, -1, 1)) + self.epsilon()
        pos_x = F.relu(x) + self.epsilon()
        log = torch.log(neg_alpha*pos_x + 1) / neg_alpha
        exp = (torch.exp(pos_alpha*pos_x) - 1) / pos_alpha
        return log + exp


activations = {'softexp': SoftExponential,
               'blu': BLU,
               'aplu': APLU,
               'prelu': PReLU,
               'pelu': PELU,
               'relu': nn.ReLU,
               'gelu': nn.GELU,
               'rrelu': nn.RReLU,
               'hardtanh': nn.Hardtanh,
               'relu6': nn.ReLU6,
               'sigmoid': nn.Sigmoid,
               'hardsigmoid': nn.Hardsigmoid,
               'tanh': nn.Tanh,
               'hardswish': nn.Hardswish,
               'elu': nn.ELU,
               'celu': nn.CELU,
               'selu': nn.SELU,
               'hardshrink': nn.Hardshrink,
               'leakyrelu': nn.LeakyReLU,
               'logsigmoid': nn.LogSigmoid,
               'softplus': nn.Softplus,
               'softshrink': nn.Softshrink,
               'softsign': nn.Softsign,
               'tanhshrink': nn.Tanhshrink}

if __name__ == '__main__':
    x = torch.zeros(5, 10)
    for name, act in activations.items():
        print(name)
        f = act()
        assert x.shape == f(x).shape
