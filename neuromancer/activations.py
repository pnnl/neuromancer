# machine learning/data science imports
import torch
import torch.nn as nn


class SoftExponential(nn.Module):
    """
    Implementation of soft exponential activation: https://arxiv.org/pdf/1602.01321.pdf
    """

    def __init__(self, alpha=0.0):
        """

        :param alpha:
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        if self.alpha == 0.0:
            return x
        elif self.alpha < 0.0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha
        else:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


class BLU(nn.Module):
    """
    Implementation of Bendable Linear Units: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8913972
    """
    def __init__(self, tune_alpha=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.epsilon = 1e-7 if tune_alpha else 0.0
        self.epsilon = nn.Parameter(torch.tensor(self.epsilon), requires_grad=False)

    def forward(self, x):
        return self.beta * (torch.sqrt(x * x + self.alpha * self.alpha + 1e-7) - self.alpha) + x


if __name__ == '__main__':
    x = BLU()
    y = SoftExponential()
    z = torch.zeros(5, 10)

    print(x(z).shape)
    print(y(z).shape)