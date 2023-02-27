import math

import torch
from torch import nn

from .butterfly_multiply import butterfly_mult, butterfly_mult_untied


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 2).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False, tied_weight=True, increasing_stride=True, ortho_init=False):
        super().__init__()
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        size = self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.nstack = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.tied_weight = tied_weight
        self.increasing_stride = increasing_stride
        self.ortho_init = ortho_init
        twiddle_core_shape = (self.nstack, size - 1) if tied_weight else (self.nstack, m, size // 2)
        if not ortho_init:
            twiddle_shape = twiddle_core_shape + ((2, 2) if not complex else (2, 2, 2))
            scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
            self.twiddle = nn.Parameter(torch.randn(twiddle_shape) * scaling)
        else:
            if not complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, (twiddle_core_shape), dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                self.twiddle = nn.Parameter(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                                         torch.stack((s, c), dim=-1)), dim=-1))
            else:
                # Sampling from the Haar measure on U(2) is a bit subtle.
                # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.randn((3, ) + twiddle_core_shape) * math.pi * 2
                A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
                B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
                C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
                D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
                self.twiddle = nn.Parameter(torch.stack((torch.stack((A, B), dim=-2),
                                                         torch.stack((C, D), dim=-2)), dim=-2))
        if bias:
            bias_shape = (out_size, ) if not complex else (out_size, 2)
            self.bias = nn.Parameter(torch.Tensor(bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, in_size) if real or (batch, in_size, 2) if complex
        Return:
            output: (batch, out_size) if real or (batch, out_size, 2) if complex
        """
        batch = input.shape[0]
        output = input
        if self.in_size != self.in_size_extended:  # Zero-pad
            padded_shape = (batch, self.in_size_extended - self.in_size) + (() if not self.complex else (2, ))
            output = torch.cat((output, torch.zeros(padded_shape, dtype=output.dtype, device=output.device)),
                               dim=-1 if not self.complex else -2)
        output = butterfly_mult(self.twiddle, output, self.increasing_stride) if self.tied_weight else butterfly_mult_untied(self.twiddle, output, self.increasing_stride)
        output = output.view((batch, self.nstack * self.in_size_extended) + (() if not self.complex else (2, )))
        out_size_extended = 1 << (int(math.ceil(math.log2(self.out_size))))
        if (self.in_size_extended // out_size_extended >= 2):  # Average instead of just take the top rows
            if not self.complex:
                output = output.view(batch, self.in_size_extended // out_size_extended, out_size_extended).mean(dim=1)
            else:
                output = output.view(batch, self.in_size_extended // out_size_extended, out_size_extended, 2).mean(dim=1)
        if self.out_size != out_size_extended:  # Take top rows
            output = output[:, :self.out_size]
        return output if self.bias is None else output + self.bias

    def extra_repr(self):
        return 'in_size={}, out_size={}, bias={}, complex={}, tied_weight={}, increasing_stride={}, ortho_init={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.complex, self.tied_weight, self.increasing_stride, self.ortho_init
        )

