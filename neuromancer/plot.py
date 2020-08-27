"""

"""

# machine learning/data science imports
import numpy as np
import scipy.linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pyts.image as pytsimg
import pyts.multivariate.image as pytsmvimg


def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return:
    """
    phi = np.linspace(0, 2 * np.pi, k)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def plot_matrices(matrices, labels, figname):
    rows = len(matrices)
    cols = len(matrices[0])
    fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(matrices[i][j])
            axes[i, j].title.set_text(labels[i][j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(figname)


def pltPhase(X, figname=None):
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
        ax = fig.gca(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
    elif X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1])
        plt.plot(X[0, 0], X[0, 1], 'ro')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
    plt.tight_layout()
    plt.show()
    if figname is not None:
        plt.savefig(figname)


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
        axes[row, col].set_title('Recurrence plot')
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
        axes[row, col].set_title('Markov Transition Field')
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
        axes[row, col].set_title('Gramian Angular Fields')
        axes[row, col].set_xlabel('X norm discretized')
        axes[row, col].set_ylabel('X norm discretized')
    plt.tight_layout()
    plt.show()


def plot_traj(data, figname=None):
    """

    :param data: dictionary
    :param figname: string
    :return:
    """
    plot_setup = [(notation, array) for
                  notation, array in data.items()]
    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    for j, (notation, array) in enumerate(plot_setup):
        ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_xlabel('Time', fontsize=20)
        ax[j, 0].set_ylabel(notation, fontsize=20)
        ax[j, 0].tick_params(axis='x', labelsize=18)
        ax[j, 0].tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def pltCL(Y, R=None, U=None, D=None, X=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset
    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='--'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and R is not None:
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(R[:, k], '--', linewidth=3, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k])
                ax[j, 0].legend(custom_lines, ['Reference', 'Output'])
                ax[j, 0].plot(Ymin[:, k], '--', linewidth=3, c='k') if Ymin is not None else None
                ax[j, 0].plot(Ymax[:, k], '--', linewidth=3, c='k') if Ymax is not None else None
        if notation == 'U':
            for k in range(array.shape[1]):
                ax[j, 0].plot(array, linewidth=3)
                ax[j, 0].plot(Umin[:, k], '--', linewidth=3, c='k') if Umin is not None else None
                ax[j, 0].plot(Umax[:, k], '--', linewidth=3, c='k') if Umax is not None else None
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=20)
        ax[j, 0].set_xlabel('Time', fontsize=20)
        ax[j, 0].set_ylabel(notation, fontsize=20)
        ax[j, 0].tick_params(axis='x', labelsize=18)
        ax[j, 0].tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


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
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=3, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=20)
        ax[j, 0].set_xlabel('Time', fontsize=20)
        ax[j, 0].set_ylabel(notation, fontsize=20)
        ax[j, 0].tick_params(axis='x', labelsize=18)
        ax[j, 0].tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)


def plot_trajectories(traj1, traj2, labels, figname):
    fig, ax = plt.subplots(len(traj1), 1, figsize=(12, 12))
    for row, (t1, t2, label) in enumerate(zip(traj1, traj2, labels)):
        if t2 is not None:
            ax[row].plot(t1, label=f'True')
            ax[row].plot(t2, '--', label=f'Pred')
        else:
            ax[row].plot(t1)
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    ylabel=label,
                    xlim=(0, len(t1)))
        ax[row].tick_params(labelbottom=False)
        ax[row].axvspan(2016, 4032, facecolor='grey', alpha=0.25, zorder=-100)
        ax[row].axvspan(4032, 6048, facecolor='grey', alpha=0.5, zorder=-100)
    ax[row].tick_params(labelbottom=True)
    ax[row].set_xlabel('Day')
    ax[0].text(64, 30, '             Train                ',
            bbox={'facecolor': 'white', 'alpha': 0.5})
    ax[0].text(2064, 30, '           Validation           ',
            bbox={'facecolor': 'grey', 'alpha': 0.25})
    ax[0].text(4116, 30, '              Test                ',
               bbox={'facecolor': 'grey', 'alpha': 0.5})
    plt.tight_layout()
    plt.savefig(figname)


def trajectory_movie(true_traj, pred_traj, figname='traj.mp4', freq=1, fps=15, dpi=100):
    plt.style.use('dark_background')
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Trajectory Movie', artist='Matplotlib',
                    comment='Demo')
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=1000)
    fig, ax = plt.subplots(len(true_traj), 1)
    true, pred = [], []
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax if len(true_traj) == 1 else ax[row]
        axe.set(xlim=(0, t1.shape[0]),
                    ylim=(min(t1.min(), t2.min()) - 0.1, max(t1.max(), t2.max()) + 0.1))
        axe.set_ylabel(label, rotation=0, labelpad=20)
        t, = axe.plot([], [], label='True', c='c')
        p, = axe.plot([], [], label='Pred', c='m')
        true.append(t)
        pred.append(p)
        axe.tick_params(labelbottom=False)
    axe.tick_params(labelbottom=True)
    axe.set_xlabel('Time')
    axe.legend()
    plt.tight_layout()
    with writer.saving(fig, figname, dpi=dpi):
        for k in range(len(true_traj[0])):
            if k % freq == 0:
                for j in range(len(true_traj)):
                    true[j].set_data(range(k), true_traj[j][:k])
                    pred[j].set_data(range(k), pred_traj[j][:k])
                writer.grab_frame()


class Animator:

    def __init__(self, dynamics_model):
        self.model = dynamics_model
        plt.style.use('dark_background')
        self.fig, (self.eigax, self.matax) = plt.subplots(1, 2)

        self.eigax.set_title('State Transition Matrix Eigenvalues')
        self.eigax.set_ylim(-1.1, 1.1)
        self.eigax.set_xlim(-1.1, 1.1)
        self.eigax.set_aspect(1)

        self.matax.axis('off')
        self.matax.set_title('State Transition Matrix')
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=15, metadata=dict(artist='Aaron Tuor'), bitrate=1800)
        self.ims = []

    def _find_mat(self, module):
        try:
            mat = module.effective_W().detach().cpu().numpy()
            if len(set(mat.shape)) != 1:
                raise AttributeError
            return mat
        except AttributeError:
            for modus in module.children():
                return self.find_mat(modus)
            return np.eye(4)

    def find_mat(self, model):
        try:
            mat = model.fx.effective_W().detach().cpu().numpy()
            if len(set(mat.shape)) != 1:
                raise AttributeError
            return mat
        except AttributeError:
            return self._find_mat(model)

    def __call__(self):
        mat = self.find_mat(self.model)
        w, v = LA.eig(mat)
        self.ims.append([self.matax.imshow(mat),
                         self.eigax.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))])

    def make_and_save(self, filename):
        eig_ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, repeat_delay=3000)
        eig_ani.save(filename, writer=self.writer)