from neuromancer.blocks import MLP
import torch.nn as nn
import torch
import slim


class JacobianMLP(nn.Module):
    def __init__(self,  insize, outsize, bias=False,
                 Linear=slim.Linear, nonlin=nn.ReLU, hsizes=[64], linargs=dict()):

        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.linear = nn.ModuleList([Linear(sizes[k],
                                            sizes[k+1],
                                            bias=bias,
                                            **linargs) for k in range(self.nhidden+1)])

    def get_jacobian(self, x, xprev):
        grads = []
        for i in range(x.shape[-1]):
            grads.append(torch.autograd.grad(x.flatten()[i], xprev, retain_graph=True)[0].flatten())
        return torch.stack(grads).T

    def forward(self, x):
        """
        Data Jacobian constructed from equation (9)
        from https://people.ece.uw.edu/bilmes/p/mypubs/wang-dnn-jacobian-2016.pdf
        :param x:
        :return:
        """
        jacobians = []
        xprev = x
        for lin, nlin in zip(self.linear, self.nonlin):
            x = lin(xprev)
            jacobians.append(self.get_jacobian(x, xprev))
            xprev = x
            x = nlin(xprev)
            jacobians.append(self.get_jacobian(x, xprev))
            xprev = x
        DJM = torch.chain_matmul(*jacobians)
        return x, DJM, jacobians


# Faster to calculate the DJM this way
x = torch.tensor([[-1.0, 1.0, 3.0, 5.0]], requires_grad=True)
fx = MLP(4, 4, nonlin=nn.ReLU, hsizes=[4, 64, 128, 4], bias=False)
grads = []
for i in range(4):
    grads.append(torch.autograd.grad(fx(x).flatten()[i], x)[0].flatten())
DJM = torch.stack(grads).T
print(fx(x))
print(torch.matmul(x, DJM))

# Sanity check to view the factorized DJM
fxj = JacobianMLP(4, 4, nonlin=nn.ReLU, hsizes=[4, 64, 28, 4])
output, DJM, jacobians = fxj(x)
print(output)
print(torch.matmul(x, DJM))
