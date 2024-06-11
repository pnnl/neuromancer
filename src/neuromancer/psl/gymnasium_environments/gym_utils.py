import torch


def sin_cos_to_t(sin_cos, positive=False):
    """
    Convert sin and cos to angle in radians
    :param sin_cos: sin and cos values
    :param positive: if True, the angle will be positive
    :return: angle in radians
    """
    sin, cos = sin_cos[..., 0], sin_cos[..., 1]
    t = torch.atan2(sin, cos)
    if positive:
        t[t < 0] = t[t < 0] + 2 * torch.pi
    return t


def bound(x, m, M=None):
    """
    Bound a value between m and M
    Farama Implementation: https://github.com/Farama-Foundation/Gymnasium/blob/52b6878618cf54ef1133342e4e34bb37d0122511/gymnasium/envs/classic_control/acrobot.py#L403
    :param x: value
    :param m: minimum value
    :param M: maximum value
    :return: bounded value
    """
    if M is None:
        M = m[1]
        m = m[0]
    return torch.clamp(x, min=m, max=M)


def wrap(x, m, M):
    """
    Wrap a value between m and M
    Farama Implementation: https://github.com/Farama-Foundation/Gymnasium/blob/52b6878618cf54ef1133342e4e34bb37d0122511/gymnasium/envs/classic_control/acrobot.py#L382
    :param x: value
    :param m: minimum value
    :param M: maximum value
    :return: wrapped value
    """

    diff = M - m
    return ((x - m) % diff) + m


def stack_k(kx):
    """
    Auxiliary function to stack the kx values for the RK4 method
    :param kx: list of kx values
    :return: stacked kx values
    """
    kx_tensor = torch.stack(kx[:-1])
    kx_tensor = torch.cat((kx_tensor, torch.tensor([kx[-1]])))
    return kx_tensor


def rk4(derivs, y0, t):
    """
    DIRECTLY FROM FARAMA FOUNDATION'S GYMNASIUM LINKED BELOW:
    https://github.com/Farama-Foundation/Gymnasium/blob/52b6878618cf54ef1133342e4e34bb37d0122511/gymnasium/envs/classic_control/acrobot.py#L422
    __________________________________________________________

    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    dt = t[1] - t[0]
    dt2 = dt / 2.0

    y = y0
    x = y[:-1]
    u = y[-1]
    k1 = derivs(None, x, u)
    k1 = stack_k(k1)
    y = y0 + dt2 * k1
    x = y[:-1]
    u = y[-1]
    k2 = derivs(None, x, u)
    k2 = stack_k(k2)
    y = y0 + dt2 * k2
    x = y[:-1]
    u = y[-1]
    k3 = derivs(None, x, u)
    k3 = stack_k(k3)
    y = y0 + dt * k3
    x = y[:-1]
    u = y[-1]
    k4 = derivs(None, x, u)
    k4 = stack_k(k4)
    yout = (y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))[:4]
    return yout


def two_D_rk4(derivs, y0, t):
    """
    Integrate a system of ODEs using 4-th order Runge-Kutta in PyTorch, separating state and control.
    :param derivs: The derivative of the system, function with signature `dy = derivs(yi, control, ti)`.
    :param y0: Initial state vector(s) with control as the last element, can be a 1-D tensor or 2-D tensor (stack of 1-D tensors).
    :param t: Sample times, 1-D tensor.
    :return: Runge-Kutta approximation of the ODE(s), 2-D tensor if y0 is 2-D, otherwise 1-D. Control is removed from the output.
    """
    if len(y0.shape) < 2:
        y0 = y0.unsqueeze(0)
    youts = []
    # for every dim call regukar rk4
    for i in range(y0.shape[0]):
        youts.append(rk4(derivs, y0[i], t))
    yout = torch.stack(youts, dim=0)
    return yout  # Return as is if input was 2-D
