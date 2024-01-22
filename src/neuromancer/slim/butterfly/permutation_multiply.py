import math

import torch
from torch import nn


use_extension = True
try:
    from factor_multiply import permutation_factor_even_odd_multiply, permutation_factor_even_odd_multiply_backward
    from factor_multiply import permutation_factor_reverse_multiply, permutation_factor_reverse_multiply_backward
except:
    use_extension = False
    import warnings
    warnings.warn("C++/CUDA extension isn't installed. Will use butterfly multiply implemented in Pytorch, which is much slower.")


def permutation_mult_torch(prob, input, increasing_stride=False, return_intermediates=False):
    """Multiply by permutation factors, parameterized by the probabilities.
    Parameters:
        prob: (nsteps, 3), where prob[:, 0] is the probability of separating the even and odd indices,
            and prob[:, 1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
            Note that stride starts at 4, not 2 (as permutations do nothing at stride 2).
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 4, 8, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 4).
            Note that this only changes the order of multiplication, not how prob is stored.
            In other words, prob[@log_stride - 1] always stores the probability for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    nsteps = prob.shape[0]
    assert prob.shape == (nsteps, 3)
    output = input.contiguous()
    intermediates = [output]
    if input.dim() == 2:  # real
        for log_stride in range(1, nsteps + 1) if increasing_stride else range(1, nsteps + 1)[::-1]:
            stride = 1 << log_stride
            # First step: weighted mean of identity permutation and permutation that yields [even, odd]
            output = ((1 - prob[log_stride - 1, 0]) * output.view(-1, 2, stride) + prob[log_stride - 1, 0] * output.view(-1, stride, 2).transpose(-1, -2))
            # Second step: weighted mean of identity permutation and permutation that reverses the first and the second half
            output = (((1 - prob[log_stride - 1, 1:]).unsqueeze(-1) * output + prob[log_stride - 1, 1:].unsqueeze(-1) * output.flip(-1)))
            intermediates.append(output)
        return output.view(batch_size, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, n) for intermediate in intermediates])
    else:  # complex
        for log_stride in range(1, nsteps + 1) if increasing_stride else range(1, nsteps + 1)[::-1]:
            stride = 1 << log_stride
            # First step: weighted mean of identity permutation and permutation that yields [even, odd]
            output = ((1 - prob[log_stride - 1, 0]) * output.view(-1, 2, stride, 2) + prob[log_stride - 1, 0] * output.view(-1, stride, 2, 2).transpose(-2, -3))
            # Second step: weighted mean of identity permutation and permutation that reverses the first and the second half
            output = (((1 - prob[log_stride - 1, 1:]).unsqueeze(-1).unsqueeze(-1) * output + prob[log_stride - 1, 1:].unsqueeze(-1).unsqueeze(-1) * output.flip(-2)))
            intermediates.append(output)
        return output.view(batch_size, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, n, 2) for intermediate in intermediates])


class PermutationFactorEvenOddMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        """Multiply by a single permutation factor that separates the even and the odd (with weight).
        Parameters:
            p: real number between 0.0 and 1.0
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            output: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        ctx.save_for_backward(p, input)
        return permutation_factor_even_odd_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            d_p: real number
            d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_even_odd_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_even_odd_mult = PermutationFactorEvenOddMult.apply


class PermutationFactorReverseMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        """Multiply by a single permutation factor that reverses the first and second halves (with weights).
        Parameters:
            p: (2, ), must be between 0.0 and 1.0
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            output: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        ctx.save_for_backward(p, input)
        return permutation_factor_reverse_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            d_p: real number
            d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_reverse_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_reverse_mult = PermutationFactorReverseMult.apply


def permutation_mult_factors(prob, input, increasing_stride=False, return_intermediates=False):
    """Multiply by permutation factors, parameterized by the probabilities.
    Parameters:
        prob: (nsteps, 3), where prob[:, 0] is the probability of separating the even and odd indices,
            and prob[:, 1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
            Note that stride starts at 4, not 2 (as permutations do nothing at stride 2).
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 4, 8, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 4).
            Note that this only changes the order of multiplication, not how prob is stored.
            In other words, prob[@log_stride - 1] always stores the probability for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    nsteps = prob.shape[0]
    assert prob.shape == (nsteps, 3)
    output = input.contiguous()
    intermediates = [output]
    if input.dim() == 2:  # real
        for log_stride in range(1, nsteps + 1) if increasing_stride else range(1, nsteps + 1)[::-1]:
            stride = 1 << log_stride
            output = output.view(batch_size * n // (2 * stride), 2 * stride)
            output = permutation_factor_even_odd_mult(prob[log_stride - 1, :1], output)
            output = permutation_factor_reverse_mult(prob[log_stride - 1, 1:], output)
            intermediates.append(output)
        return output.view(batch_size, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, n) for intermediate in intermediates])
    else:  # complex
        for log_stride in range(1, nsteps + 1) if increasing_stride else range(1, nsteps + 1)[::-1]:
            stride = 1 << log_stride
            output = output.view(batch_size * n // (2 * stride), 2 * stride, 2)
            output = permutation_factor_even_odd_mult(prob[log_stride - 1, :1], output)
            output = permutation_factor_reverse_mult(prob[log_stride - 1, 1:], output)
            intermediates.append(output)
        return output.view(batch_size, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, n, 2) for intermediate in intermediates])


permutation_mult = permutation_mult_factors if use_extension else permutation_mult_torch


def permutation_mult_single_factor_torch(prob, input):
    """Multiply by a single permutation factor.
    Parameters:
        prob: (3, ), where prob[0] is the probability of separating the even and odd indices,
            and prob[1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    assert prob.shape == (3, )
    output = input.contiguous()
    if input.dim() == 2:  # real
        stride = n // 2
        # First step: weighted mean of identity permutation and permutation that yields [even, odd]
        output = ((1 - prob[0]) * output.view(-1, 2, stride) + prob[0] * output.view(-1, stride, 2).transpose(-1, -2))
        # Second step: weighted mean of identity permutation and permutation that reverses the first and the second half
        output = (((1 - prob[1:]).unsqueeze(-1) * output + prob[1:].unsqueeze(-1) * output.flip(-1)))
        return output.view(batch_size, n)
    else:  # complex
        stride = n // 2
        # First step: weighted mean of identity permutation and permutation that yields [even, odd]
        output = ((1 - prob[0]) * output.view(-1, 2, stride, 2) + prob[0] * output.view(-1, stride, 2, 2).transpose(-2, -3))
        # Second step: weighted mean of identity permutation and permutation that reverses the first and the second half
        output = (((1 - prob[1:]).unsqueeze(-1).unsqueeze(-1) * output + prob[1:].unsqueeze(-1).unsqueeze(-1) * output.flip(-2)))
        return output.view(batch_size, n, 2)


def permutation_mult_single_factor(prob, input):
    """Multiply by a single permutation factor, parameterized by the probabilities.
    Parameters:
        prob: (3, ), where prob[0] is the probability of separating the even and odd indices,
            and prob[1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    assert prob.shape == (3, )
    output = input.contiguous()
    output = permutation_factor_even_odd_mult(prob[:1], output)
    output = permutation_factor_reverse_mult(prob[1:], output)
    return output


permutation_mult_single = permutation_mult_single_factor if use_extension else permutation_mult_single_factor_torch
