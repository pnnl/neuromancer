"""
Recurrent Neural Network implementation for use with structured linear maps.
"""
# machine learning/data science imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ecosystem imports
import neuromancer.slim as slim


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, nonlin=F.gelu,
                 hidden_map=slim.Linear, input_map=slim.Linear, input_args=dict(),
                 hidden_args=dict()):
        """

        :param input_size: (int) Dimension of input to rnn cell.
        :param hidden_size: (int) Dimension of output of rnn cell.
        :param bias: (bool) Whether to use bias.
        :param nonlinearity: (callable) Activation function
        :param linear_map: (nn.Module) A module compatible with torch.nn.Linear
        :param linargs: (dict) Arguments to instantiate linear layers

        """
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlin
        self.lin_in = input_map(input_size, hidden_size, bias=bias, **input_args)
        self.lin_hidden = hidden_map(hidden_size, hidden_size, bias=bias, **hidden_args)

    def reg_error(self):
        """

        :return: (torch.float) Regularization error associated with linear maps.
        """
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        """

        :param input: (torch.Tensor, shape=[batch_size, input_size]) Input to cell
        :param hidden: (torch.Tensor, shape=[batch_size, hidden_size]) Hidden state (typically previous output of cell)
        :return: (torch.Tensor, shape=[batchsize, hidden_size]) Cell output

        .. doctest::

            >>> import neuromancer.slim as slim, torch
            >>> cell = slim.RNNCell(5, 8, input_map=slim.Linear, hidden_map=slim.PerronFrobeniusLinear)
            >>> x, h = torch.rand(20, 5), torch.rand(20, 8)
            >>> output = cell(x, h)
            >>> output.shape
            torch.Size([20, 8])

        """

        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, cell_args=dict()):
        """
        Has input and output corresponding to basic usage of torch.nn.RNN module.
        No bidirectional, bias, nonlinearity, batch_first, and dropout args.
        Cells can incorporate custom linear maps.
        Bias and nonlinearity are included in cell args.

        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together
                                  to form a stacked RNN, with the second RNN taking in outputs of the first RNN and
                                  computing the final results. Default: 1
        :param cell_args: (dict) Arguments to instantiate RNN cells (see :class:`rnn.RNNCell` for args).
        """
        super().__init__()
        rnn_cells = [RNNCell(input_size, hidden_size, **cell_args)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, **cell_args)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.init_states = nn.ParameterList([nn.Parameter(torch.zeros(1, cell.hidden_size))
                                             for cell in self.rnn_cells])

    def reg_error(self):
        """

        :return: (torch.float) Regularization error associated with linear maps.
        """
        return torch.mean(torch.stack([cell.reg_error() for cell in self.rnn_cells]))

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_state: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`, initial hidden states for stacked RNNCells
        :returns:

            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.

        .. doctest::

            >>> import neuromancer.slim as slim, torch
            >>> rnn = slim.RNN(5, hidden_size=8, num_layers=3, cell_args={'hidden_map': slim.PerronFrobeniusLinear})
            >>> x = torch.rand(20, 10, 5)
            >>> output, h_n = rnn(x)
            >>> output.shape, h_n.shape
            (torch.Size([20, 10, 8]), torch.Size([3, 10, 8]))

        """
        assert len(sequence.shape) == 3, f'RNN takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        if init_states is None:
            init_states = self.init_states
        final_hiddens = []
        for h, cell in zip(init_states, self.rnn_cells):
            # loop over stack of cells
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                # loop over sequence
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hiddens.append(h)  # Save final hidden state for each cell in case need to do truncated back prop
        assert torch.equal(sequence[-1, :, :], final_hiddens[-1])
        return sequence, torch.stack(final_hiddens)


if __name__ == '__main__':
    x = torch.rand(20, 5, 8)

    for bias in [True, False]:
        for num_layers in [1, 2]:
            for name, map in slim.maps.items():
                print(name)
                print(map)
                rnn = RNN(8, hidden_size=8, num_layers=num_layers, cell_args={'bias': bias,
                                                                              'nonlin': F.gelu,
                                                                              'hidden_map': map,
                                                                              'input_map': slim.Linear})
                out = rnn(x)
                print(out[0].shape, out[1].shape)

            for map in set(slim.maps.values()) - slim.square_maps:
                print(name)
                rnn = RNN(8, hidden_size=16, num_layers=num_layers, cell_args={'bias': bias,
                                                                               'nonlin': F.gelu,
                                                                               'hidden_map': map,
                                                                               'input_map': slim.Linear})
                out = rnn(x)
                print(out[0].shape, out[1].shape)
