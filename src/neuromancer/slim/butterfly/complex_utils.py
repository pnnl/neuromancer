''' Utility functions for handling complex tensors: conjugate and complex_mul.
Pytorch (as of 1.0) does not support complex tensors, so we store them as
float tensors where the last dimension is 2 (real and imaginary parts).
'''

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

# For now, it seems that the overhead of converting to cupy makes the cupy
# version slower than the torch version. So I'll just disable cupy.

# # Check if cupy is available
# if torch.cuda.is_available():
#     use_cupy = True
#     try:
#         import cupy as cp
#     except:
#         use_cupy = False
#         print("Cupy isn't installed or isn't working properly. Will use Pytorch's complex multiply, which is slower.")
# else:
#     use_cupy = False
use_cupy = False


def torch2numpy(X):
    """Convert a torch float32 tensor to a numpy array, sharing the same memory.
    """
    return X.detach().numpy()


def torch2cupy(tensor):
    return cp.fromDlpack(to_dlpack(tensor.cuda()))


def cupy2torch(tensor):
    return from_dlpack(tensor.toDlpack())


def real_to_complex(X):
    """A version of X that's complex (i.e., last dimension is 2).
    Parameters:
        X: (...) tensor
    Return:
        X_complex: (..., 2) tensor
    """
    return torch.stack((X, torch.zeros_like(X)), dim=-1)


def conjugate_torch(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


class Conjugate(torch.autograd.Function):
    '''X is a complex64 tensors but stored as float32 tensors, with last dimension = 2.
    '''
    @staticmethod
    def forward(ctx, X):
        assert X.shape[-1] == 2, 'Last dimension must be 2'
        if X.is_cuda:
            if use_cupy:
                # TODO: do we need .contiguous here? I think it doesn't work if the last dimension isn't contiguous
                return cupy2torch(torch2cupy(X).view('complex64').conj().view('float32'))
            else:
                return conjugate_torch(X)
        else:
            return torch.from_numpy(np.ascontiguousarray(torch2numpy(X)).view('complex64').conj().view('float32'))

    @staticmethod
    def backward(ctx, grad):
        return Conjugate.apply(grad)


conjugate = Conjugate.apply


def complex_mul_torch(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def complex_mul_numpy(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64')
    Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64')
    return torch.from_numpy((X_np * Y_np).view('float32'))


class ComplexMul(torch.autograd.Function):
    '''X and Y are complex64 tensors but stored as float32 tensors, with last dimension = 2.
    '''
    @staticmethod
    def forward(ctx, X, Y):
        assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
        ctx.save_for_backward(X, Y)
        if X.is_cuda:
            assert Y.is_cuda, 'X and Y must both be torch.cuda.FloatTensor'
            if use_cupy:
                # TODO: do we need .contiguous here? I think it doesn't work if the last dimension isn't contiguous
                return cupy2torch((torch2cupy(X).view('complex64') * torch2cupy(Y).view('complex64')).view('float32'))
            else:
                return complex_mul_torch(X, Y)
        else:
            assert not Y.is_cuda, 'X and Y must both be torch.FloatTensor'
            X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64')
            Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64')
            return torch.from_numpy((X_np * Y_np).view('float32'))

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = ComplexMul.apply(grad, conjugate(Y)), ComplexMul.apply(grad, conjugate(X))
        # Need to sum over dimensions that were broadcasted
        dims_to_sum_X = [-i for i in range(1, X.dim() + 1) if X.shape[-i] != grad.shape[-i]]
        dims_to_sum_Y = [-i for i in range(1, Y.dim() + 1) if Y.shape[-i] != grad.shape[-i]]
        if dims_to_sum_X:  # If empty list is passed to sum, it sums all the dimensions
            grad_X = grad_X.sum(dim=dims_to_sum_X, keepdim=True)
        if dims_to_sum_Y:  # If empty list is passed to sum, it sums all the dimensions
            grad_Y = grad_Y.sum(dim=dims_to_sum_Y, keepdim=True)
        if grad.dim() > X.dim():
            grad_X = grad_X.sum(tuple(range(grad.dim() - X.dim())))
        if grad.dim() > Y.dim():
            grad_Y = grad_Y.sum(tuple(range(grad.dim() - Y.dim())))
        return grad_X, grad_Y


complex_mul = ComplexMul.apply


def test_complex_mul():
    n = 5
    m = 7
    p = 4
    X = torch.rand(m, 1, n, 2, requires_grad=True).transpose(0, 2)  # Transpose to test non-contiguous arrays
    Y = torch.rand(m, p, 2, requires_grad=True).transpose(0, 1)
    Z = complex_mul(X, Y)
    Z_torch = complex_mul_torch(X, Y)
    assert Z.shape == (n, p, m, 2)
    assert torch.allclose(Z, Z_torch)
    g = torch.rand_like(Z)
    dX, dY = torch.autograd.grad(Z, (X, Y), g)
    dX_torch, dY_torch = torch.autograd.grad(Z_torch, (X, Y), g)
    assert torch.allclose(dX, dX_torch)
    assert torch.allclose(dY, dY_torch)
    if torch.cuda.is_available():
        X, Y = X.cuda(), Y.cuda()
        Z = complex_mul(X, Y)
        Z_torch = complex_mul_torch(X, Y)
        assert Z.shape == (n, p, m, 2)
        assert torch.allclose(Z, Z_torch)
        g = torch.rand_like(Z)
        dX, dY = torch.autograd.grad(Z, (X, Y), g)
        dX_torch, dY_torch = torch.autograd.grad(Z_torch, (X, Y), g)
        assert torch.allclose(dX, dX_torch)
        assert torch.allclose(dY, dY_torch)


def complex_matmul_torch(X, Y):
    """Multiply two complex matrices.
    Parameters:
       X: (..., n, m, 2)
       Y: (..., m, p, 2)
    Return:
       Z: (..., n, p, 2)
    """
    return complex_mul_torch(X.unsqueeze(-2), Y.unsqueeze(-4)).sum(dim=-3)


class ComplexMatmulNp(torch.autograd.Function):
    """Multiply two complex matrices, in numpy.
    Parameters:
        X: (n, m, 2)
        Y: (m, p, 2)
    Return:
        Z: (n, p, 2)
    """

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64').squeeze(-1)
        Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64').squeeze(-1)
        prod = torch.from_numpy((X_np @ Y_np)[..., None].view('float32'))
        return prod

    @staticmethod
    def backward(ctx, grad):
        X, Y  = ctx.saved_tensors
        X_np = X.detach().contiguous().numpy().view('complex64').squeeze(-1)
        Y_np = Y.detach().contiguous().numpy().view('complex64').squeeze(-1)
        grad_np = grad.detach().contiguous().numpy().view('complex64').squeeze(-1)
        dX = torch.from_numpy(np.expand_dims(grad_np @ Y_np.conj().T, -1).view('float32'))
        dY = torch.from_numpy(np.expand_dims(X_np.conj().T @ grad_np, -1).view('float32'))
        return dX, dY


complex_matmul = ComplexMatmulNp.apply


def test_complex_mm():
    n = 5
    m = 7
    p = 4
    X = torch.rand(n, m, 2, requires_grad=True)
    Y = torch.rand(m, p, 2, requires_grad=True)
    Z = complex_matmul(X, Y)
    assert Z.shape == (n, p, 2)
    batch_size = 3
    # X = torch.rand(batch_size, n, m, 2)
    # Y = torch.rand(batch_size, m, p, 2)
    # Z = complex_matmul(X, Y)
    # assert Z.shape == (batch_size, n, p, 2)
    X_np = X.detach().contiguous().numpy().view('complex64').squeeze(-1)
    Y_np = Y.detach().contiguous().numpy().view('complex64').squeeze(-1)
    Z_np = np.expand_dims(X_np @ Y_np, axis=-1).view('float32')
    assert np.allclose(Z.detach().numpy(), Z_np)
    Z_torch = complex_matmul_torch(X, Y)
    assert torch.allclose(Z, Z_torch)
    g = torch.rand_like(Z)
    dX, dY = torch.autograd.grad(Z, (X, Y), g)
    dX_torch, dY_torch = torch.autograd.grad(Z_torch, (X, Y), g)
    assert torch.allclose(dX, dX_torch)
    assert torch.allclose(dY, dY_torch)


if __name__ == '__main__':
    test_complex_mul()
    test_complex_mm()
