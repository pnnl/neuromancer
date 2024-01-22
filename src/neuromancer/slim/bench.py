"""
Several toy problems adapted from:

+ https://github.com/zmhammedi/Orthogonal_RNN/blob/master/toy_problems.py

.. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory." Neural
computation 9.8 (1997): 1735-1780.
.. [2] Ilya Sutskever et al. "On the importance of initialization and momentum
in deep learning." Proceedings of the 30th international conference on machine
learning (ICML-13). 2013.
.. [3] Herbert Jaegar. "Long Short-Term Memory in Echo State Networks: Details
of a Simulation Study." Jacobs University Technical Report No. 27, 2012.
.. [4] James Martens and Ilya Sutskever. "Learning recurrent neural networks
with hessian-free optimization." Proceedings of the 28th International
Conference on Machine Learning (ICML-11). 2011.
.. [5] Quoc V. Le, Navdeep Jaitly, and Geoffrey E. Hinton. "A Simple Way to
Initialize Recurrent Networks of Rectified Linear Units." arXiv preprint
arXiv:1504.00941 (2015).

Various toy problems which are meant to test whether a model can learn
long-term dependencies.  Originally proposed in [1]_, with variants used more
recently in [2]_, [3]_, [4]_, etc.  In general, we follow the descriptions in
[1]_ because they are the most comprehensive; variants of the original tasks
have been used instead in the cited papers.

"""
import numpy as np
import functools
from mnist import MNIST
import six.moves.cPickle as cPickle
import os


def gen_masked_sequences(rng, T, n_sequences, sample):
    # Sample the noise dimension
    noise_dim = sample(size=(n_sequences, T, 1))
    # Initialize mask dimension to all zeros
    mask_dim = np.zeros((n_sequences, T, 1))

    N_1 = rng.choice(range(T - 1), n_sequences)
    N_2 = rng.choice(range(T - 1), n_sequences)

    # If N_1 = N_2 for any sequences, add 1 to avoid
    N_2[N_2 == N_1] = N_2[N_2 == N_1] + 1

    # Set the add indices to 1
    mask_dim[np.arange(n_sequences), N_1] = 1
    mask_dim[np.arange(n_sequences), N_2] = 1

    # Concatenate noise and mask dimensions to create data
    X = np.concatenate([noise_dim, mask_dim], axis=-1)
    return X


def ackley(rng, nsamples, dim=2):
    X = rng.uniform(low=-40., high=40., size=(nsamples, dim))
    k = np.sqrt(np.mean(np.square(X), axis=1))
    j = np.mean(np.cos(2.*np.pi*X), axis=1)

    return X, -(20. + np.e - 20.*np.exp(k) - np.exp(j))


def add(rng, T, n_sequences):
    X = gen_masked_sequences(rng, T, n_sequences, functools.partial(rng.uniform, high=1., low=0.))
    # Sum the entries in the third dimension where the second is 1
    y = np.sum((X[:, :, 0]*(X[:, :, 1] == 1)), axis=1)
    return X, y


def multiply(rng, T, n_sequences):
    """
    Generate sequences and target values for the "multiply" task, as
    described in [1]_ section 5.5.  Sequences are two dimensional where the
    first dimension are values sampled uniformly at random from [0, 1] and the
    second dimension is either -1, 0, or 1: At the first and last steps, it is
    -1; at one of the first ten steps (``N_1``) it is 1; and at a step between
    0 and ``.5*min_length`` (``N_2``) it is also 1.  The goal is to predict
    ``X_1*X_2`` where ``X_1`` and ``X_2`` are the values of the first dimension
    at ``N_1`` and ``N_2`` respectively.  For example, the target for the
    following sequence
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      | -1  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  | -1  |``
    would be ``.3*.9 = .27``.
    Parameters
    ----------
    T : int
        Sequence length.
    n_sequences : int
        Number of sequences to generate.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    References
    ----------
    .. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    """
    X = gen_masked_sequences(rng, T, n_sequences, functools.partial(rng.uniform, high=1., low=0.))

    # Sum the entries in the third dimension where the second is 1
    y = np.prod((X[:, :, 0]*(X[:, :, 1] == 1)), axis=1)
    return X, y


def xor(rng, T, n_sequences):
    """ Generate sequences and target values for the "XOR" task, as
    described in [1]_ section 4.1.  Sequences are two dimensional where the
    first dimension are binary values sampled uniformly at random from {0, 1}
    and the second dimension is either -1, 0, or 1: At the first and last
    steps, it is -1; at one of the first ten steps (``N_1``) it is 1; and at a
    step between 0 and ``.5*min_length`` (``N_2``) it is also 1.  The goal is
    to predict ``X_1^X_2`` where ``X_1`` and ``X_2`` are the values of the
    first dimension at ``N_1`` and ``N_2`` respectively.  For example, the
    target for the following sequence
    ``|  1 | 0 | 1 | 0 | 0 | ... | 1 | 1 | ... | 0 |  0 |
      | -1 | 0 | 1 | 0 | 0 |     | 0 | 1 |     | 0 | -1 |``
    would be ``1^1 = 0``.
    Parameters
    ----------
    T : int
        Sequence length.
    n_sequences : int
        Number of sequences to generate.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    References
    ----------
    .. [1] James Martens and Ilya Sutskever. "Learning recurrent neural
    networks with hessian-free optimization." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.
    """
    # Get sequences
    X, mask = gen_masked_sequences(rng, T, n_sequences, functools.partial(rng.choice, a=[0, 1]))
    # X[:, :, 1] > 0 constructs a boolean matrix of the rows/columns which have
    # a 1 in the last dimension of X.  X[X[:, :, 1] > 0, 0] then masks the
    # entries of the random bit dimension accordingly.  The reshape converts
    # the resulting array back into a matrix, where entries are picked in
    # "fortran" order which allows them to be correctly reshaped to (2,
    # n_sequences).  Finally, the * uses the first dimension as the arguments
    # to logical_xor
    y = np.logical_xor(*np.reshape(X[X[:, :, 1] > 0, 0], (2, n_sequences), 'F'))
    return X, y


def load_pMNIST_data(rng, data_path, perm=False):
    data_path += '/MNIST/python-mnist/data/'
    mndata = MNIST(data_path)
    tmp_train =  mndata.load_training()
    tmp_test =  mndata.load_testing()
    len_data = len(tmp_train[0])
    ind_train = rng.permutation(len_data)[:55000]
    ind_valid = rng.permutation(len_data)[55000:]
    train_set = [1. / 256 * np.asarray(tmp_train[0])[ind_train], np.asarray(tmp_train[1])[ind_train],
                 784 * np.ones(55000)]
    valid_set = [1. / 256 * np.asarray(tmp_train[0])[ind_valid], np.asarray(tmp_train[1])[ind_valid],
                 784 * np.ones(5000)]
    test_set = [1. / 256 * np.asarray(tmp_test[0]), np.asarray(tmp_test[1]),
                784 * np.ones(len(tmp_test[0])).astype('int32')]
    if os.path.isfile('prm'):
        with open('prm', 'rb') as file:
            prm = cPickle.load(file)
        file.close()
    else:
        prm = rng.permutation(784)
        with open('prm', 'wb') as file:
            cPickle.dump(prm, file, cPickle.HIGHEST_PROTOCOL)
        file.close()
    if perm:
        train_set[0] = train_set[0][:, prm]
        valid_set[0] = valid_set[0][:, prm]
        test_set[0] = test_set[0][:, prm]
    print('Data loaded.')
    return train_set, valid_set, test_set
