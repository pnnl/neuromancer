"""
# TODO: stream plots for phase spaces of ODEs
# TODO: generate correlation network - https://python-graph-gallery.com/327-network-from-correlation-matrix/
# TODO: plot information-theoretic measures for time series data - https: // elife - asu.github.io / PyInform / timeseries.html


"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyts.image as pytsimg
import pyts.multivariate.image as pytsmvimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return:
    """
    phi = np.linspace(0, 2 * np.pi, k)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def pltPhase(X, Xtrain=None, figname=None):
    """
    plot phase space for 2D and 3D state spaces

    https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/plot_streamplot.html
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.streamplot.html
    https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.quiver.html
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/
    http://systems-sciences.uni-graz.at/etextbook/sw2/phpl_python.html
    """
    fig = plt.figure()
    if X.shape[1] >= 3:
        ax = fig.add_subplot(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2])
        if Xtrain is not None:
            ax.plot(Xtrain[:, 0], Xtrain[:, 1], Xtrain[:, 2], '--')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
    elif X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1])
        plt.plot(X[0, 0], X[0, 1], 'ro')
        if Xtrain is not None:
            plt.plot(Xtrain[:, 0], Xtrain[:, 1], '--')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
    plt.show()


def pltCorrelate(X, figname=None):
    """
    plot correlation matrices of time series data
    https://realpython.com/numpy-scipy-pandas-correlation-python/
    """
    #  Pearson product-moment correlation coefficients.
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
    C = np.corrcoef(X.T)
    im1 = axes[0, 0].imshow(C)
    axes[0, 0].set_title('Pearson correlation coefficients')
    axes[0, 0].set_xlabel('$X$')
    axes[0, 0].set_ylabel('$X$')
    # covariance matrix
    C = np.cov(X.T)
    im2 = axes[0, 1].imshow(C)
    axes[0, 1].set_title('Covariance matrix')
    axes[0, 1].set_xlabel('$X$')
    axes[0, 1].set_ylabel('$X$')
    #  Spearman correlation coefficient
    rho, pval = stats.spearmanr(X, X)
    C = rho[0:X.shape[1], 0:X.shape[1]]
    im3 = axes[0, 2].imshow(C)
    axes[0, 2].set_title('Spearman correlation coefficients')
    axes[0, 2].set_xlabel('$X$')
    axes[0, 2].set_ylabel('$X$')
    plt.tight_layout()
    plt.show()
    if figname is not None:
        plt.savefig(figname)


def pltRecurrence(X, figname=None):
    """
    plot recurrence of time series data
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_rp.html
    https://pyts.readthedocs.io/en/stable/auto_examples/multivariate/plot_joint_rp.html#sphx-glr-auto-examples-multivariate-plot-joint-rp-py
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_mtf.html
    https://arxiv.org/pdf/1610.07273.pdf
    https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py
    """
    size = np.ceil(np.sqrt(X.shape[1])).astype(int)
    row_off = size-np.ceil(X.shape[1]/size).astype(int)
    # Recurrence plot
    rp = pytsimg.RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1,X.shape[1]+1):
        row = (np.ceil(i/size)-1).astype(int)
        col = (i-1)%size
        C = X_rp[i-1]
        im = axes[row, col].imshow(C)
        axes[row, col].set_title(f'Recurrence plot x_{i}')
        axes[row, col].set_xlabel('time')
        axes[row, col].set_ylabel('time')
    plt.tight_layout()
    plt.show()

    # joint recurrence plot
    jrp = pytsmvimg.JointRecurrencePlot(threshold='point', percentage=50)
    X_jrp = jrp.fit_transform(X.T.reshape(X.shape[1], 1, -1))
    fig = plt.figure()
    C = X_jrp[0]
    plt.imshow(C)
    plt.title('joint recurrence plot')
    plt.xlabel('time')
    plt.ylabel('time')
    plt.tight_layout()
    plt.show()

    # Markov Transition Field
    mtf = pytsimg.MarkovTransitionField(image_size=100)
    X_mtf = mtf.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1, X.shape[1] + 1):
        row = (np.ceil(i / size) - 1).astype(int)
        col = (i - 1) % size
        C = X_mtf[i - 1]
        axes[row, col].imshow(C)
        axes[row, col].set_title(f'Markov Transition Field x_{i}')
        axes[row, col].set_xlabel('X norm discretized')
        axes[row, col].set_ylabel('X norm discretized')
    plt.tight_layout()
    plt.show()

    # Gramian Angular Fields
    gasf = pytsimg.GramianAngularField(image_size=100, method='summation')
    X_gasf = gasf.fit_transform(X.T)
    fig, axes = plt.subplots(nrows=size-row_off, ncols=size, squeeze=False)
    for i in range(1, X.shape[1] + 1):
        row = (np.ceil(i / size) - 1).astype(int)
        col = (i - 1) % size
        C = X_gasf[i - 1]
        im = axes[row, col].imshow(C)
        axes[row, col].set_title(f'Gramian Angular Fields x_{i}')
        axes[row, col].set_xlabel('X norm discretized')
        axes[row, col].set_ylabel('X norm discretized')
    plt.tight_layout()
    plt.show()


def pltOL(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            colors = get_colors(array.shape[1]+1)
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=2, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=2, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=2)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def pltCL(Y, U=None, D=None, X=None, R=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y':
            if R is not None:
                colors = get_colors(array.shape[1]+1)
                for k in range(array.shape[1]):
                    ax[j, 0].plot(array[:, k], '-', linewidth=2, c=colors[k])
                ax[j, 0].plot(R, '--', linewidth=2, c='black')
                ax[j, 0].legend(custom_lines, ['Ref', 'Y'])
            else:
                ax[j, 0].plot(array, linewidth=2)
            if Ymax is not None:
                ax[j, 0].plot(Ymax, '--', linewidth=2, c='red')
            if Ymin is not None:
                ax[j, 0].plot(Ymin, '--', linewidth=2, c='red')
        else:
            ax[j, 0].plot(array, linewidth=2)
            if notation == 'U' and Umax is not None:
                ax[j, 0].plot(Umax, '--', linewidth=2, c='red')
            if notation == 'U' and Umin is not None:
                ax[j, 0].plot(Umin, '--', linewidth=2, c='red')
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
