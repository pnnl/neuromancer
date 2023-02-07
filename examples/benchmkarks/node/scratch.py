import numpy as np
def truncated_mse(true, pred):
    diffsq = (true - pred) ** 2
    truncs = diffsq > 1.0
    print(~truncs*diffsq)
    print(truncs * np.ones_like(diffsq))
    tmse = truncs * np.ones_like(diffsq) + ~truncs * diffsq
    return tmse.mean()

x = np.array([1., 2., 2, 5, 8])
y = np.array([1.5, 5, 800, 5.1, 8.2])

print(truncated_mse(x, y))