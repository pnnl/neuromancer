import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult, butterfly_mult_factors, butterfly_mult_inplace

batch_size = 256
n = 1024
B = Butterfly(n, n, bias=False).to('cuda')
L = torch.nn.Linear(n, n, bias=False).to('cuda')
x = torch.randn(batch_size, n, requires_grad=True).to('cuda')
twiddle = B.twiddle
# twiddle = torch.randn(2, 2, n - 1, device=x.device, requires_grad=True).permute(2, 0, 1)


import time
nsteps = 1000
# nsteps = 1

grad = torch.randn_like(x)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult_factors(twiddle.squeeze(0), x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult factors forward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult factors backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult_factors(twiddle.squeeze(0), x)
    torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult factors together: {end - start}s')

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult_inplace(twiddle.squeeze(0), x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult in-place forward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult in-place backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult_inplace(twiddle.squeeze(0), x)
    torch.autograd.grad(output, (twiddle, x), grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult in-place together: {end - start}s')

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult(twiddle, x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult intermediate forward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (twiddle, x), grad.unsqueeze(1), retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult intermediate backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_mult(twiddle, x)
    torch.autograd.grad(output, (twiddle, x), grad.unsqueeze(1))
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly mult intermediate together: {end - start}s')

# output = x @ W.t()  # Do it once so that cuBlas handles are initialized, etc.
output = L(x)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = L(x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm forward: {end - start}s')
torch.autograd.grad(output, (L.weight, x), grad, retain_graph=True)
output.backward(gradient=grad, retain_graph=True)  # Do it once just to be safe
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (L.weight, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = L(x)
    torch.autograd.grad(output, (L.weight, x), grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm together: {end - start}s')

output = torch.rfft(x, 1)  # Do it once so that cuFFT plans are cached, etc.
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = torch.rfft(x, 1)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT forward: {end - start}s')
grad = torch.randn_like(output)  # Do it once just to be safe
torch.autograd.grad(output, x, grad, retain_graph=True)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, x, grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = torch.rfft(x, 1)
    torch.autograd.grad(output, x, grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT together: {end - start}s')

# output = B(x)
# output.backward(gradient=grad)
# output = L(x)
# output.backward(gradient=grad)
# output = torch.rfft(x, 1)
# output = butterfly_mult_inplace(twiddle, x)
# output.backward(gradient=grad)
# output = butterfly_mult(twiddle, x)
# torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)
# output.backward(gradient=grad)
# B = Block2x2DiagProduct(n, complex=True).to('cuda')
# x = torch.randn(batch_size, n, 2, requires_grad=True).to('cuda')
# twiddle = torch.randn_like(twiddle_list_concat(B), requires_grad=True)
# output = butterfly_mult(twiddle, x)
# grad = torch.randn_like(output)
# output.backward(gradient=grad)
# torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)


# x = torch.randn(3)
# w_init = torch.randn(3)
# w = torch.tensor(w_init, requires_grad=True)

# optimizer = torch.optim.SGD([w], lr=0.1)
# for i in range(10):
#     optimizer.zero_grad()
#     loss = x @ w
#     loss.backward()
#     optimizer.step()
# loss = x @ w
# print(loss.item())

# k = 2.0
# w_new = torch.tensor(w_init / k, requires_grad=True)
# optimizer = torch.optim.SGD([w_new], lr=0.1 / k**2)
# for i in range(10):
#     optimizer.zero_grad()
#     loss = x @ (k * w_new)
#     loss.backward()
#     optimizer.step()
# loss = x @ (k * w_new)
# print(loss.item())
