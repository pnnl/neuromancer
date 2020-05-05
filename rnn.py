# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports
import linear

# TODO: shall we merge rnn.py with blocks.py?
#  I would vote for the same I/O properties as blocks
# TODO: RNN cells are returning zero dimension tensor with reg_error
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=F.gelu, Linear=linear.Linear, **linargs):
        super().__init__()
        # TODO: uniform notation
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlinearity
        self.lin_in = Linear(input_size, hidden_size, bias=bias, **linargs)
        self.lin_hidden = Linear(hidden_size, hidden_size, bias=bias, **linargs)
        if type(Linear) is linear.Linear:
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
        self.in_features, self.out_features = input_size, hidden_size
        rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                     Linear=Linear, **linargs)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                      Linear=Linear, **linargs)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.num_layers = len(rnn_cells)
        self.init_states = nn.ParameterList([nn.Parameter(torch.zeros(1, cell.hidden_size)) for cell in self.rnn_cells])

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
        for h, cell in zip(self.init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h.unsqueeze(0))
            sequence = torch.cat(states, 0)
            final_hiddens.append(h)
        final_hiddens = torch.cat(final_hiddens, 0)
        # return sequence # TODO: temporary fix
        return sequence, final_hiddens

if __name__ == '__main__':
    x = torch.rand(20, 5, 7)
    for map in linear.maps:
        rnn = RNN(7, 7, num_layers=1, Linear=map)
        out = rnn(x)
        print(out[0].shape, out[1].shape)

    for map in set(linear.maps) - linear.square_maps:
        rnn = RNN(7, 64, num_layers=1, Linear=map)
        out = rnn(x)
        print(out[0].shape, out[1].shape)

    for map in linear.maps:
        rnn = RNN(7, 7, num_layers=5, Linear=map)
        out = rnn(x)
        print(out[0].shape, out[1].shape)

    for map in set(linear.maps) - linear.square_maps:
        rnn = RNN(7, 64, num_layers=5, Linear=map)
        out = rnn(x)
        print(out[0].shape, out[1].shape)
