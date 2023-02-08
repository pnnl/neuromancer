from koopman import Autoencoder, StateInclusive, INN
from torch import nn
import torch

for lift in [Autoencoder, StateInclusive, INN]:
    print(lift)
    inn = lift(5, 11, bias=True, linear_map=nn.Linear,
               nonlin=nn.ELU, hsizes=[28 for l in range(3)])

    x = torch.randn(12, 5)
    z = inn(x)
    print(x.shape)
    print(z.shape)

    xprime = inn(z, rev=True)
    print(xprime.shape)


for lift in [INN, StateInclusive, Autoencoder]:
    inn = lift(2, 10, bias=True, linear_map=nn.Linear,
               nonlin=nn.ELU, hsizes=[28 for l in range(3)])

    x = torch.randn(1, 2, 2)
    z = inn(x)
    print(x)

    xprime = inn(z, rev=True)
    print(xprime)

    print(torch.nn.functional.mse_loss(x, xprime))