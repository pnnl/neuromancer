"""
Another way to solve: with bounds on variables
bnds = ((0.4, 0.6), (1.0, 10.0), (0.0, 30.0))
solution = minimize(objective,x0,bounds=bnds,method='SLSQP')

####
Dataset splits

Train:
All data
Set 1: 4 PID, 4 relay, 2 constant power

Individual
Set 2: 4 PID
Set 3: 4 relay
Set 4: 2 constant power

Ablation
Set 5: 4 PID, Relay
Set 6: 4 relay, constant power
Set 7: 2 constant power, PID
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy import genfromtxt
from neuromancer.datasets import MultiExperimentDataset, systems


noisy_pid_idxs = [1, 8]
offkilter_pid_idx = [5]
train_pid_idxs = [4]
constant_idxs = [6, 7]
train_relay_idxs = [10, 11, 12, 14]
all_train = set(train_pid_idxs + constant_idxs + train_relay_idxs)

all_dev_exp, all_test_exp = [3, 9], [2, 13]
dev_exp, test_exp = [13], [9]
# dev_exp, test_exp = [3], [13]
all_pid = [3, 4, 5, 8]
all_pid_dev = [1]
all_pid_test = [2]

datasplits = {'all': {'train': list(all_train),
                      'dev': dev_exp,
                      'test': test_exp},
              'pid': {'train': train_pid_idxs,
                      'dev': dev_exp,
                      'test': test_exp},
              'constant': {'train': constant_idxs,
                           'dev': dev_exp,
                           'test': test_exp},
              'relay': {'train': train_relay_idxs,
                        'dev': dev_exp,
                        'test': test_exp},
              'no_pid': {'train': list(all_train - set(train_pid_idxs)),
                         'dev': dev_exp,
                         'test': test_exp},
              'no_constant': {'train': list(all_train - set(constant_idxs)),
                              'dev': dev_exp,
                              'test': test_exp},
              'no_relay': {'train': list(all_train - set(train_relay_idxs)),
                           'dev': dev_exp,
                           'test': test_exp},
              'all_pid': {'train': all_pid,
                          'dev': all_pid_dev,
                          'test': all_pid_test}}


def prep_data(data):
    """

    :param data: (np.array, shape=(ns,3))
    :return:
    """
    print(data.shape)
    u0 = data[0, 1]
    yp0 = data[0, 2]
    t = data[:, 0].T - data[0, 0]
    print(data[:, 0].T.shape, data[0, 0].shape)
    u = data[:, 1].T
    yp = data[:, 2].T
    ns = len(t)  # specify number of steps
    delta_t = t[1]-t[0]
    print(t.shape, u.shape)
    uf = interp1d(t, u)  # create linear interpolation of the u data versus time
    return u0, yp0, t, u, yp, ns, delta_t, uf


def merge_data(data):
    y0 = data['Yp'][:, :, 0].numpy().transpose()
    u0 = data['Up'][:, :, 0].numpy().transpose()
    print('y', y0.max(), y0.min())
    print('u', u0.max(), u0.min())
    t = np.array([[0.1124*k for k in range(y0.shape[1])] for j in range(y0.shape[0])])
    model_data = np.stack([t, u0, y0], axis=2)
    return model_data


def fopdt(y, t, uf, Km, taum, thetam, u0, yp0):
    """
    first-order plus dead-time approximation

    :param y: output
    :param t: time
    :param uf: input linear function (for time shift)
    :param Km: model gain
    :param taum: model time constant
    :param thetam: model time delay
    :param u0:
    :param yp0:
    :return: dydt
    """
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        um = u0
    dydt = (-(y-yp0) + Km * (um-u0))/taum
    return dydt


# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x, u0, yp0, t, ns, uf):
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    ym = np.zeros(ns)
    # initial condition
    ym[0] = yp0
    for i in range(0, ns-1):
        ts = [t[i], t[i+1]]
        y1 = odeint(fopdt, ym[i], ts, args=(uf, Km, taum, thetam, u0, yp0))
        ym[i+1] = y1[-1]
    ym = np.array(ym)
    return ym


def evaluate(x, dataset):
    pass


if __name__ == '__main__':
    # Import CSV data file
    # Column 1 = time (t)
    # Column 2 = input (u)
    # Column 3 = output (yp)
    # data = genfromtxt('data.txt', delimiter=',')  # shape=(51,3)
    # print(data.shape)

    # Import dataset via neuromancer loading
    split = datasplits['relay']
    dataset = dataset = MultiExperimentDataset(system='fsw_phase_2', nsim=10000000,
                                     norm=[], nsteps=1, savedir='test',
                                     split=split)
    train_data = merge_data(dataset.train_data)
    test_data = merge_data(dataset.test_loop[0])
    print(train_data.shape, test_data.shape)
    x = np.array([103.53, 100.0, 1.8])
    # # # show initial objective
    # # print('Initial SSE Objective: ' + str(multi_objective(x0, train_data)))
    #
    # # # optimize Km, taum, thetam
    # # solution = minimize(lambda x: multi_objective(x, train_data), x0)
    # # x = solution.x
    #
    # # show final objective
    # print('Final SSE Objective: ' + str(multi_objective(x, train_data)))
    #
    # print('Kp: ' + str(x[0]))
    # print('taup: ' + str(x[1]))
    # print('thetap: ' + str(x[2]))
    #
    # # calculate model with updated parameters
    test_data = test_data[0]#[240:458]
    test_data[:, 0] = np.array([0.1124*k for k in range(len(test_data))])
    test_data[:, 1] = (test_data[:, 1] - np.mean(test_data[:, 1]))/np.std(test_data[:, 1])
    test_data[:, 2] = (test_data[:, 2] - np.mean(test_data[:, 2]))/np.std(test_data[:, 2])
    u0, yp0, t, u, yp, ns, delta_t, uf = prep_data(test_data)
    # ym1 = sim_model(x0, u0, yp0, t, ns, uf)
    ym2 = sim_model(x, u0, yp0, t, ns, uf)

    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    print(yp.shape)
    print(t.shape, yp.shape)
    plt.plot(yp, linewidth=2, label='Process Data')
    # plt.plot(t, ym1, 'b-', linewidth=2, label='Initial Guess')
    plt.plot(ym2, 'r--', linewidth=3, label='Optimized FOPDT')
    plt.ylabel('Output')
    plt.legend(loc='best')
    plt.savefig('fsw_fopdt.png')
    # plt.subplot(2, 1, 2)
    # plt.plot(t, u, 'bx-', linewidth=2)
    # plt.plot(t, uf(t), 'r--', linewidth=3)
    # plt.legend(['Measured', 'Interpolated'], loc='best')
    # plt.ylabel('Input Data')
    # plt.show()