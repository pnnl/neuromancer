# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports
import linear


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu, Linear=linear.Linear, **linargs):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = Linear(input_size, hidden_size, bias=bias, **linargs)
        self.lin_hidden = Linear(hidden_size, hidden_size, bias=bias, **linargs)
        if type(Linear) is linear.Linear():
            torch.nn.init.orthogonal_(self.lin_hidden.linear.weight)

    def reg_error(self):
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=False, nonlinearity=F.gelu, Linear=linear.Linear, **linargs):
        """

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param nonlinearity:
        :param stable:
        """
        super().__init__()
        cell = RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                       Linear=Linear, **linargs)
        rnn_cells = [cell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)]
        rnn_cells += [cell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.num_layers = len(rnn_cells)
        self.init_state = nn.Parameter(torch.zeros(self.num_layers, 1, rnn_cells[0].hidden_size))

    def reg_error(self):
        return torch.mean(torch.stack([cell.reg_error() for cell in self.rnn_cells]))

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
