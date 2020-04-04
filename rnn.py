import torch
import torch.nn as nn
import torch.nn.functional as F
from linear import SpectralLinear, PerronFrobeniusLinear, SVDLinear, ConstrainedLinear


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hidden = nn.Linear(hidden_size, hidden_size, bias=bias)
        torch.nn.init.kaiming_uniform_(self.lin_in.weight)
        torch.nn.init.eye_(self.lin_hidden.weight)
        if bias:
            torch.nn.init.zeros_(self.lin_in.bias)

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class SSMCell(nn.Module):
    """Cells all return 2 state values for common API with LSTM"""
    def __init__(self, nx, nu, nd, ny, stable=True, bias=False):
        super().__init__()
        if stable:
            self.A = ConstrainedLinear(nx, nx)
        else:
            self.A = nn.Linear(nx, nx, bias=bias)
        self.B = nn.Linear(nu, nx, bias=bias)
        self.E = nn.Linear(nd, nx, bias=bias)
        self.C = nn.Linear(nx, ny, bias=bias)
        self.hidden_size = nx + ny

        with torch.no_grad():
            self.C.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

        for param in self.C.parameters():
            param.requires_grad = False

    def forward(self, u_d, x_y):
        u, d = u_d[:, :1], u_d[:, 1:]
        x, y = x_y[:, :4], x_y[:, 4:]
        x = self.A(x) + self.B(u) + self.E(d)
        y = self.C(x)
        x_y = torch.cat([x, y], dim=1)
        return x_y


class PerronFrobeniusCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = ConstrainedLinear(input_size, hidden_size, bias=bias, init='identity')
        self.lin_hidden = ConstrainedLinear(hidden_size, hidden_size, bias=bias, init='identity')

    def forward(self, input, hidden):
        return self.nonlin(.5*self.lin_hidden(hidden) + .5*self.lin_in(input))


class SpectralCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = nn.Linear(input_size, hidden_size, bias=bias)
        torch.nn.init.kaiming_uniform_(self.lin_in.weight)
        torch.nn.init.eye_(self.lin_hidden.weight)
        self.lin_hidden = SpectralLinear(hidden_size, hidden_size, bias=bias)
        if bias:
            torch.nn.init.zeros_(self.lin_in.bias)

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class SVDCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = SVDLinear(input_size, hidden_size, bias=bias)
        torch.nn.init.kaiming_uniform_(self.lin_in.weight)
        torch.nn.init.eye_(self.lin_hidden.weight)
        self.lin_hidden = SVDLinear(hidden_size, hidden_size, bias=bias)
        if bias:
            torch.nn.init.zeros_(self.lin_in.bias)

    def spectral_error(self):
        return self.lin_in.spectral_error + self.lin_hidden.spectral_error

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=False, nonlinearity=F.gelu, cell=RNNCell):
        """

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param nonlinearity:
        :param stable:
        """
        super().__init__()
        rnn_cells = [cell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)]
        rnn_cells += [cell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.num_layers = len(rnn_cells)
        self.init_state = nn.Parameter(torch.zeros(self.num_layers, 1, rnn_cells[0].hidden_size))

    def forward(self, sequence):
        """
        :param sequence: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: h_0 (num_layers, batch, hidden_size)
        :returns:
        - output: (seq_len, batch, hidden_size)
        - h_n: (num_layers, batch, hidden_size)
        """
        final_hiddens = []
        for h, cell in zip(self.init_state, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h.unsqueeze(0))
            states = torch.cat(states, 0)
            final_hiddens.append(h)
        final_hiddens = torch.cat(final_hiddens, 0)
        return states, final_hiddens


class SVDRNN(RNN):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=False, nonlinearity=F.gelu):
        super().__init__(input_size, hidden_size, num_layers=num_layers, bias=bias,
                         nonlinearity=nonlinearity, cell=SVDCell)

    @property
    def spectral_error(self):
        return torch.sum(torch.cat([cell.spectral_error for cell in self.RNNCells]))


if __name__ == '__main__':
    mb = 1
    sl = 100
    nx = 4
    nh = 2

    model1 = RNN(nx, nh, num_layers=60, nonlinearity=F.gelu, stable=True, bias=True)
    model2 = torch.nn.RNN(nx, nh, num_layers=60, nonlinearity='relu', bias=True)
    inputs = torch.randn(sl, mb, nx)**2 + 20
    print(inputs)
    output, hiddens = model1(inputs)
    output2, hiddens2 = model2(inputs)
    print(output[-1])
    print(output2[-1])
