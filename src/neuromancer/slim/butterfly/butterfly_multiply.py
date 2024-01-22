import math

import torch
from torch import nn

from .complex_utils import complex_mul

use_extension = True
try:
    from factor_multiply import butterfly_multiply_intermediate, butterfly_multiply_intermediate_backward
    from factor_multiply import butterfly_multiply_untied, butterfly_multiply_untied_backward
    from factor_multiply import butterfly_multiply_inplace, butterfly_multiply_inplace_backward
    from factor_multiply import butterfly_factor_multiply, butterfly_factor_multiply_backward
except:
    use_extension = False
    # import warnings
    # warnings.warn("C++/CUDA extension isn't installed. Will use butterfly multiply implemented in Pytorch, which is much slower.")


def butterfly_mult_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 2).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    nstack = twiddle.shape[0]
    assert twiddle.shape == (nstack, n - 1, 2, 2) if input.dim() == 2 else (nstack, n - 1, 2, 2, 2)
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    if input.dim() == 2:  # real
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n)
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1)  # shape (nstack, 2, 2, stride)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t.unsqueeze(1) * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n) for intermediate in intermediates])
    else:  # complex
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n, 2)
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1, 4)  # shape (nstack, 2, 2, stride, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t.unsqueeze(1), output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n, 2) for intermediate in intermediates])


class ButterflyMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 2).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        output_and_intermediate = butterfly_multiply_intermediate(twiddle, input, increasing_stride)
        ctx.save_for_backward(twiddle, output_and_intermediate)
        ctx._increasing_stride = increasing_stride
        return output_and_intermediate[-1]

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        twiddle, output_and_intermediate = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        d_coefficients, d_input = butterfly_multiply_intermediate_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None  # Autograd requires 3 gradients

butterfly_mult = ButterflyMult.apply if use_extension else butterfly_mult_torch


def butterfly_mult_untied_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 2).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    nstack = twiddle.shape[0]
    assert twiddle.shape == (nstack, m, n // 2, 2, 2) if input.dim() == 2 else (nstack, m, n // 2, 2, 2, 2)
    if input.dim() == 2:  # real
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n)
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)  # shape (nstack, n // (2 * stride, )2, 2, stride)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n) for intermediate in intermediates])
    else:  # complex
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n, 2)
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2, 2).permute(0, 1, 3, 4, 2, 5)  # shape (nstack, n // (2 * stride, )2, 2, stride, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t, output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n, 2) for intermediate in intermediates])


class ButterflyMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log 2, n / 2, 2, 2) if real or (nstack, log 2, n / 2, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 2).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        output_and_intermediate = butterfly_multiply_untied(twiddle, input, increasing_stride)
        ctx.save_for_backward(twiddle, output_and_intermediate)
        ctx._increasing_stride = increasing_stride
        return output_and_intermediate[-1]

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, log 2, n / 2, 2, 2) if real or (nstack, log 2, n / 2, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, log 2, n / 2, 2, 2) if real or (nstack, log 2, n / 2, 2, 2, 2) if complex
            d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        twiddle, output_and_intermediate = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        d_coefficients, d_input = butterfly_multiply_untied_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None  # Autograd requires 3 gradients

butterfly_mult_untied = ButterflyMultUntied.apply if use_extension else butterfly_mult_untied_torch


class ButterflyMultInplace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """Experimental in-place implementation that does not store intermediate results.
        Instead, the intermediate results are computed from the output during the backward pass.
        Parameters:
            twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 2).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        assert increasing_stride, 'Decreasing stride not implemented'
        output = butterfly_multiply_inplace(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_multiply_inplace_backward(grad, twiddle, output)
        return d_coefficients, d_input

butterfly_mult_inplace = ButterflyMultInplace.apply


class ButterflyFactorMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        """Multiply by a single factor.
        Parameters:
            twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
            input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        Returns:
            output: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        """
        ctx.save_for_backward(twiddle, input)
        return butterfly_factor_multiply(twiddle, input)

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        Returns:
            d_twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
            d_input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        """
        twiddle, input = ctx.saved_tensors
        d_twiddle, d_input = butterfly_factor_multiply_backward(grad, twiddle, input)
        return d_twiddle, d_input

butterfly_factor_mult = ButterflyFactorMult.apply


def butterfly_mult_factors(twiddle, input, increasing_stride=True, return_intermediates=False):
    """Implementation that have separate kernels for each factor, for debugging.
    Parameters:
        twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 2, 4, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 2).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    assert twiddle.shape == (n - 1, 2, 2) if input.dim() == 2 else (n - 1, 2, 2, 2)
    output = input.contiguous()
    intermediates = [output]
    if input.dim() == 2:  # real
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[(stride - 1):(2 * stride - 1)].permute(1, 2, 0)  # shape (2, 2, stride)
            output_reshape = output.view(batch_size * n // (2 * stride), 2, stride)
            output = butterfly_factor_mult(t, output_reshape)
            intermediates.append(output)
        return output.view(batch_size, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, n) for intermediate in intermediates])
    else:  # complex
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[(stride - 1):(2 * stride - 1)].permute(1, 2, 0, 3)  # shape (2, 2, stride, 2)
            output_reshape = output.view(batch_size * n // (2 * stride), 2, stride, 2)
            output = butterfly_factor_mult(t, output_reshape)
            intermediates.append(output)
        return output.view(batch_size, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, n, 2) for intermediate in intermediates])
