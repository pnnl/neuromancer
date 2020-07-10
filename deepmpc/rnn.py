# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports
import linear


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, nonlin=F.gelu, Linear=linear.Linear, linargs=dict()):
        """

        :param input_size:
        :param hidden_size:
        :param bias:
        :param nonlinearity:
        :param Linear:
        :param linargs:
        """
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlin
        self.lin_in = Linear(input_size, hidden_size, bias=bias, **linargs)
        self.lin_hidden = Linear(hidden_size, hidden_size, bias=bias, **linargs)
        if type(Linear) is linear.Linear:
            torch.nn.init.orthogonal_(self.lin_hidden.linear.weight)

    def reg_error(self):
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hsizes=(16,),
                 bias=False, nonlin=F.gelu, Linear=linear.Linear, linargs=dict()):
        """

        :param input_size:
        :param output_size:
        :param hsizes:
        :param bias:
        :param nonlinearity:
        :param stable:
        """
        super().__init__()
        assert len(set(hsizes)) == 1, 'All hiddens sizes should be equal for the RNN implementation'
        hidden_size = hsizes[0]
        num_layers = len(hsizes)
        self.in_features, self.out_features = input_size, hidden_size
        rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlin=nonlin,
                     Linear=Linear, linargs=linargs)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias, nonlin=nonlin,
                      Linear=Linear, linargs=linargs)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.num_layers = len(rnn_cells)
        self.init_states = nn.ParameterList([nn.Parameter(torch.zeros(1, cell.hidden_size)) for cell in self.rnn_cells])

    def reg_error(self):
        return torch.mean(torch.stack([cell.reg_error() for cell in self.rnn_cells]))

    def forward(self, sequence, init_states=None):
        """
        :param sequence: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: h_0 (num_layers, batch, hidden_size)
        :returns:
        - output: (seq_len, batch, hidden_size)
        - h_n: (num_layers, batch, hidden_size)
        """
        assert len(sequence.shape) == 3, 'RNN takes order 3 tensor with shape=(seq_len, nsamples, dim)'
        if init_states is None:
            init_states = self.init_states
        final_hiddens = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h.unsqueeze(0))
            sequence = torch.cat(states, 0)
            final_hiddens.append(h)
        final_hiddens = torch.stack(final_hiddens, 0)
        assert torch.equal(sequence[-1, :, :], final_hiddens[-1, :, :])
        return sequence, final_hiddens


if __name__ == '__main__':
    x = torch.rand(20, 5, 8)
    for bias in [True, False]:
        for map in linear.maps.values():
            rnn = RNN(8, hsizes=[8, 8], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for map in set(linear.maps.values()) - linear.square_maps:
            rnn = RNN(8, hsizes=[16, 16], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for map in linear.maps.values():
            rnn = RNN(8, hsizes=[8, 8], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for map in set(linear.maps.values()) - linear.square_maps:
            rnn = RNN(8, hsizes=[16, 16], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)
