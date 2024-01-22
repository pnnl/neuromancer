from scipy import interpolate
import numpy as np


def test_interpolate():
    x = np.random.normal(size=[100, 5])
    print(x.shape)
    y = np.arange(0, 100, dtype=np.float64)*0.1
    print(y.shape)
    func = interpolate.interp1d(y, x, axis=0, kind='previous')
    z = np.array([func(i) for i in y])
    print(z.shape)
    print(x[0], z[0])
    assert np.array_equal(x, z)