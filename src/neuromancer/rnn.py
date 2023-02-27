import torch
import torch.nn as nn
import neuromancer.slim as slim


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, nonlin=nn.GELU, linear_map=slim.Linear, linargs=dict()):
        """

        :param input_size: (int) Number of features in time series input
        :param hidden_size: (int) Number of hidden features
        :param bias: (bool) Whether to use bias
        :param nonlinearity: (class) Constructor to instantiate activation function.
        :param linear_map: (class) Constructor to instantiate linear map
        :param linargs: (class) Arguments to instantiate linear map
        """
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlin
        self.lin_in = linear_map(input_size, hidden_size, bias=bias, **linargs)
        self.lin_hidden = linear_map(hidden_size, hidden_size, bias=bias, **linargs)
        if type(linear_map) is slim.Linear:
            torch.nn.init.orthogonal_(self.lin_hidden.linear.weight)

    def reg_error(self):
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        """

        :param input: (torch.Tensor, shape=[batchsize, input_size])
        :param hidden: (torch.Tensor, shape=[batchsize, hidden_size])
        :return: (torch.Tensor, shape=[batchsize, hidden_size])
        """
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hsizes=(16,),
                 bias=False, nonlin=nn.GELU, linear_map=slim.Linear, linargs=dict()):
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
        rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlin=nonlin(),
                     linear_map=linear_map, linargs=linargs)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias, nonlin=nonlin(),
                      linear_map=linear_map, linargs=linargs)
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
        final_hiddens = final_hiddens
        assert torch.equal(sequence[-1, :, :], final_hiddens[-1])
        return sequence, torch.stack(final_hiddens)
