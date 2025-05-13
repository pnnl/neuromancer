"""
Various helper functions for plotting.
"""
from itertools import combinations

from scipy import stats
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import pyts.image as pytsimg
import pyts.multivariate.image as pytsmvimg
import torch
from matplotlib import cm
import matplotlib.image as mpimg

try:
    import pydot
except ImportError:
    pydot = None

"""

"""
# python base imports
import os

# machine learning/data science imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import matplotlib.patches as mpatches

# local imports
from neuromancer.dataset import unbatch_tensor
from neuromancer.modules import blocks

def _add_obj_components(graph, objs, components, data_keys, style="solid"):
    for obj in objs:
        graph.add_node(pydot.Node(obj.name, label=obj.name, shape="box", color="red", style=style))
        common_keys = [
            (c.name, k) for c in components for k in c.output_keys
            if k in obj.variable_names
        ] + [
            ("in", k) for k in data_keys
            if k in obj.variable_names
        ]
        for n, key in common_keys:
            graph.add_edge(pydot.Edge(n, obj.name, label=key))


def plot_model_graph(model, data_keys, include_objectives=True, fname="model_graph.png"):
    if pydot is None:
        print("Error: pydot could not be imported, could not plot model graph")
        return

    graph = pydot.Dot("model", graph_type="digraph", splines="spline", rankdir="LR")

    graph.add_node(pydot.Node("in", label="", color="white", shape="box"))

    for component in model.components:
        graph.add_node(pydot.Node(component.name, label=component.name, shape="box"))
        for key in set(component.input_keys) & set(data_keys):
            graph.add_edge(pydot.Edge("in", component.name, label=key))

    for src, dst in combinations(model.components, 2):
        common_keys = set(src.output_keys) & set(dst.input_keys)
        for key in common_keys:
            graph.add_edge(pydot.Edge(src.name, dst.name, label=key))

    if include_objectives:
        _add_obj_components(graph, model.objectives, model.components, data_keys)
        _add_obj_components(graph, model.constraints, model.components, data_keys, style="dashed")

    graph.write_png(fname)
    img = mpimg.imread(fname)
    fig = plt.imshow(img, aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def get_colors(k):
    """
    Returns k colors evenly spaced across the color wheel.
    :param k: (int) Number of colors you want.
    :return: (np.array, shape=[k, 3])
    """
    phi = np.linspace(0, 2 * np.pi, k)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    return rgb_cycle


def plot_matrices(matrices, labels, figname):
    """
    Plots and saves figure of a grid of matrices.
    Useful for inspecting layers of weights of neural networks.

    :param matrices: (list of lists of 2-way np.arrays) Grid of matrices to plot
    :param labels: (list of lists of str) Labels for plotted matrices
    :param figname: (str) Figure name ending with file extension of filetype to save as.

    .. doctest::

        >>> import neuromancer.plot as plot
        >>> color_matrices = [[get_colors(k*j) for k in range(2, 4)] for j in range(8, 11)]
        >>> labels = [[f'{k*j} X 3 matrix' for k in range(2, 4)] for j in range(8, 11)]
        >>> plot_matrices(color_matrices, labels, 'matrix_grid.png')
    """
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
    :param X: (np.array, shape=[numpoints, {2,3}])
    :param figname: (str) Filename for plot with extension for file type.

    plot phase space for 2D and 3D state spaces

    + https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/plot_streamplot.html
    + https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.streamplot.html
    + https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.quiver.html
    + http://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/
    + http://systems-sciences.uni-graz.at/etextbook/sw2/phpl_python.html

    .. doctest::

        >> import numpy as np
        >> import neuromancer.plot as plot
        >> x = np.stack([np.linspace(-10, 10, 100)]*100)
        >> y = np.stack([np.linspace(-10, 10, 100)]*100).T
        >> z = x**2 + y**2
        >> xyz = np.stack([x.flatten(), y.flatten(), z.flatten()])
        >> plot.pltPhase(xyz, figname='phase.png')
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
        plt.plot(X[-1, 0], X[-1, 1], 'g*')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
    plt.tight_layout()
    plt.show()
    if figname is not None:
        plt.savefig(figname)


def pltCorrelate(X, figname=None):
    """
    plot correlation matrices of time series data

    + https://realpython.com/numpy-scipy-pandas-correlation-python/
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

    + https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_rp.html
    + https://pyts.readthedocs.io/en/stable/auto_examples/multivariate/plot_joint_rp.html#sphx-glr-auto-examples-multivariate-plot-joint-rp-py
    + https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_mtf.html
    + https://arxiv.org/pdf/1610.07273.pdf
    + https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py
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

    :param data: (dict {str: np.array}) Dictionary of labels and time series
    :param figname: (str)
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


def pltCL(Y, R=None, U=None, D=None, X=None, ctrl_outputs=None,
          Ymin=None, Ymax=None, Umin=None, Umax=None, figname=None):
    """
    plot input output closed loop dataset

    """
    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['y', 'x', 'u', 'd'], [Y, X, U, D]) if
                  array is not None]

    controlled_y_idx = np.zeros([Y.shape[1], 1])
    controlled_y_idx[ctrl_outputs] = 1

    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)
    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='--'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='-')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'y' and R is not None:
            colors = get_colors(array.shape[1])
            for k in range(array.shape[1]):
                rk = ctrl_outputs.index(k) if ctrl_outputs is not None and k in ctrl_outputs else None
                ax[j, 0].plot(array[:, k], '-', linewidth=3, c=colors[k]) if array[:, k] is not None else None
                if rk is not None:
                    ax[j, 0].plot(R[:, rk], '--', linewidth=3, c=colors[rk]) if R[:, rk] is not None else None
                    ax[j, 0].plot(Ymin[:, rk], '--', linewidth=3, c='k') if Ymin[:, rk] is not None else None
                    ax[j, 0].plot(Ymax[:, rk], '--', linewidth=3, c='k') if Ymax[:, rk] is not None else None
                else:
                    ax[j, 0].plot(R, '--', linewidth=3, c='r') if R is not None else None
                    ax[j, 0].plot(Ymin, '--', linewidth=3, c='k') if Ymin is not None else None
                    ax[j, 0].plot(Ymax, '--', linewidth=3, c='k') if Ymax is not None else None
                # ax[j, 0].legend(custom_lines, ['Reference', 'Output'])
        if notation == 'u':
            for k in range(array.shape[1]):
                ax[j, 0].plot(array, linewidth=3)
                ax[j, 0].plot(Umin[:, k], '--', linewidth=3, c='k') if Umin is not None else None
                ax[j, 0].plot(Umax[:, k], '--', linewidth=3, c='k') if Umax is not None else None
        else:
            ax[j, 0].plot(array, linewidth=3)
        ax[j, 0].grid(True)
        # ax[j, 0].set_title(name, fontsize=22)
        ax[j, 0].set_xlabel('Time', fontsize=26)
        ax[j, 0].set_ylabel(notation, fontsize=26)
        ax[j, 0].tick_params(axis='x', labelsize=26)
        ax[j, 0].tick_params(axis='y', labelsize=26)
        ax[j, 0].set_xlim(0, Y.shape[0])
    plt.tight_layout()
    plt.show()
    if figname is not None:
        plt.savefig(figname)


def pltOL(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
    """
    plot trained open loop dataset
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
    plt.show()

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


"""
  Parametric programming example plots
"""


def plot_solution_mpp(model, xmin=-2, xmax=2, save_path=None):
    """
    plots solution landscape for problem with 2 parameters and 1 decision variable
    :param net:
    :param xmin:
    :param xmax:
    :param save_path:
    :return:
    """
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = model.net(features)
    plot_u = uu.detach().numpy()[:,:,0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), plot_u,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    ax.set(title='Solution landscape')
    if save_path is not None:
        plt.savefig(save_path+'/solution.pdf')


def plot_loss_mpp(model, dataset, xmin=-2, xmax=2, save_path=None):
    """
    plots loss function for multiparametric problem with 2 parameters
    :param model:
    :param dataset:
    :param xmin:
    :param xmax:
    :param save_path:
    :return:
    """
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    dataset_plt = dataset.dataset.get_full_batch()
    name = dataset_plt['name']
    Loss = np.ones([x.shape[0], y.shape[0]])*np.nan

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            # check loss
            X = torch.stack([x[[i]], y[[j]]]).reshape(1,1,-1)
            dataset_plt['theta'] = X
            step = model(dataset_plt)
            Loss[i,j] = step[name+'_loss'].detach().numpy()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Loss,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$L$')
    ax.set(title='Loss landscape')
    # plt.colorbar(surf)
    if save_path is not None:
        plt.savefig(save_path+'/loss.pdf')


"""
    Double Integrator DPC example plots
"""

def plot_loss_DPC(model, policy, A, B, dataset, xmin=-5, xmax=5, save_path=None):
    """
    plot loss function for trained DPC model
    :param model:
    :param dataset:
    :param xmin:
    :param xmax:
    :param save_path:
    :return:
    """
    x = torch.arange(xmin, xmax, 0.2)
    y = torch.arange(xmin, xmax, 0.2)
    xx, yy = torch.meshgrid(x, y)
    dataset_plt = dataset.dataset.get_full_batch()
    name = dataset_plt['name']
    nsteps = dataset.dataset.nsteps
    Loss = np.ones([x.shape[0], y.shape[0]])*np.nan
    # Alpha contraction coefficient: ||x_k+1|| = alpha * ||x_k||
    Alpha = np.ones([x.shape[0], y.shape[0]])*np.nan
    # ||A+B*Kx||
    Phi_norm = np.ones([x.shape[0], y.shape[0]])*np.nan
    policy = policy.net
    Anp = A.detach().numpy()
    Bnp = B.detach().numpy()
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            # check loss
            x_batch = torch.stack([x[[i]], y[[j]]]).repeat(dataset_plt['Yf'].shape[1], 1)
            X = x_batch.reshape(1, -1, A.shape[0])
            if nsteps == 1:
                dataset_plt['Yp'] = X
                dataset_plt['Yf'] = dataset_plt['Yf'][[0],:,:]
                step = model(dataset_plt)
                Loss[i,j] = step[name+'_loss'].detach().numpy()
            # check contraction
            x0 = X[:,0,:].view(1, X.shape[-1])
            Astar, bstar, _, _, _ = lpv_batched(policy, x0)
            BKx = torch.mm(B, Astar[:, :, 0])
            phi = A + BKx
            Phi_norm[i,j] = torch.norm(phi, 2).detach().numpy()
            # print(torch.matmul(Astar[:, :, 0], x0.transpose(0, 1))+bstar)
            u = policy(x0).detach().numpy()
            xnp = x0.transpose(0, 1).detach().numpy()
            xnp_n = np.matmul(Anp, xnp) + np.matmul(Bnp, u)
            if not np.linalg.norm(xnp) == 0:
                # Alpha[i,j] = np.linalg.norm(xnp_n)/np.linalg.norm(xnp)
                Alpha[i,j] = np.linalg.norm(xnp_n) - np.linalg.norm(xnp)
            else:
                Alpha[i, j] = 0

    if nsteps == 1:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Loss,
                               cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        ax.set(ylabel='$x_1$')
        ax.set(xlabel='$x_2$')
        ax.set(zlabel='$L$')
        ax.set(title='Loss landscape')
        # plt.colorbar(surf)
        if save_path is not None:
            plt.savefig(save_path+'/loss.pdf')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Phi_norm,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$Phi$')
    ax.set(title='CLS 2-norm')
    if save_path is not None:
        plt.savefig(save_path+'/phi_norm.pdf')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Alpha,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$alpha$')
    ax.set(title='CLS contraction')
    if save_path is not None:
        plt.savefig(save_path+'/contraction.pdf')

    fig1, ax1 = plt.subplots()
    cm_map = plt.cm.get_cmap('RdBu_r')
    im1 = ax1.imshow(Alpha, vmin=abs(Alpha).min(), vmax=abs(Alpha).max(),
                     cmap=cm_map, origin='lower',
                     extent=[xx.detach().numpy().min(), xx.detach().numpy().max(),
                             yy.detach().numpy().min(), yy.detach().numpy().max()],
                     interpolation="bilinear")
    fig1.colorbar(im1, ax=ax1)
    ax1.set(ylabel='$x_1$')
    ax1.set(xlabel='$x_2$')
    ax1.set(title='CLS contraction regions')
    im1.set_clim(0., 2.)  #  color limit
    if save_path is not None:
        plt.savefig(save_path+'/contraction_regions.pdf')


def plot_policy(net, xmin=-5, xmax=5, save_path=None):
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = net(features)
    plot_u = uu.detach().numpy()[:,:,0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), plot_u,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    ax.set(title='Policy landscape')
    if save_path is not None:
        plt.savefig(save_path+'/policy.pdf')


def plot_policy_train(A, B, policy, policy_list, xmin=-5, xmax=5, save_path=None):
    # Writer = animation.writers['ffmpeg']
    Writer = animation.PillowWriter
    writer = Writer(fps=5, metadata=dict(artist='Aaron Tuor'), bitrate=1800)

    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)

    U_plot = []
    for k in range(len(policy_list)):
        policy.load_state_dict(policy_list[k])
        # print(sum(sum(policy.net.linear[1].weight)))
        uu = policy.net(features)
        plot_u = uu.detach().numpy()[:, :, 0]
        U_plot.append(plot_u)

    fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    ax.set(title='Policy landscape')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(-1.05, 1.05)

    ims = []
    modulator = 5
    for i in range(len(policy_list)):
        # ttl = plt.text(xmax/2,xmax/2, 1.2, str(i), horizontalalignment='center',
        #                verticalalignment='bottom',
        #                transform=ax.transAxes)
        if i%modulator == 0:
            surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), U_plot[i],
                               cmap=cm.viridis,
                               linewidth=0, antialiased=False)
            ims.append([surf])
    ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True)
    plt.show()
    writer.fps = writer.fps/modulator
    print('saving policy landscape')
    ani.save(save_path + '/policy_3D_animation_train2.gif', writer=writer)
    # ani.save(save_path + '/policy_animation_train.gif', writer='imagemagick')


def cl_simulate(A, B, net, nstep=50, x0=np.ones([2, 1])):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    Anp = A.detach().numpy()
    Bnp = B.detach().numpy()
    x = x0
    X = [x]
    U = []
    for k in range(nstep+1):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        # taking a first control action based on RHC principle
        u = net(x_torch).detach().numpy()[:, [0]]
        x = np.matmul(Anp, x) + np.matmul(Bnp, u)
        X.append(x)
        U.append(u)
    Xnp = np.asarray(X)[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]
    return Xnp, Unp


def plot_cl(X, U, nstep=50, save_path=None, trace_movie=False):
    Umin = -1*np.ones(nstep+1)
    Umax = 1*np.ones(nstep+1)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(X, label='x', linewidth=2)
    ax[0].set(ylabel='$x$')
    ax[0].set(xlabel='time')
    ax[0].grid()
    ax[0].set_xlim(0, nstep)
    ax[1].plot(U, label='u', drawstyle='steps',  linewidth=2)
    ax[1].plot(Umin, linestyle='--', color='k', label='u', linewidth=2)
    ax[1].plot(Umax,  linestyle='--', color='k', label='u', linewidth=2)
    ax[1].set(ylabel='$u$')
    ax[1].set(xlabel='time')
    ax[1].grid()
    ax[1].set_xlim(0, nstep)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'/closed_loop_dpc.pdf')


def plot_cl_train(X_list, U_list, nstep=50, save_path=None):
    Umin = -1*np.ones(nstep+1)
    Umax = 1*np.ones(nstep+1)
    fig, ax = plt.subplots(2, 1)
    camera = Camera(fig)
    for i in range(len(X_list)):
        ax[0].plot(X_list[i][:, 0], label='x1', color='tab:blue', linewidth=2)
        ax[0].plot(X_list[i][:, 1], label='x2', color='tab:orange', linewidth=2)
        ax[0].set(ylabel='$x$')
        ax[0].set(xlabel='time')
        ax[0].grid()
        ax[0].text
        ax[0].text(31, 7, "Epoch " + str(i), fontsize=14)
        ax[0].set_xlim(0, nstep)
        ax[0].set_ylim(-10, 10)
        ax[0].set_title('Closed-loop control trajectories')
        ax[1].plot(U_list[i], label='u', color='tab:blue', drawstyle='steps', linewidth=2)
        ax[1].plot(Umin, linestyle='--', color='k', label='u', linewidth=2)
        ax[1].plot(Umax, linestyle='--', color='k', label='u', linewidth=2)
        ax[1].set(ylabel='$u$')
        ax[1].set(xlabel='time')
        ax[1].grid()
        ax[1].set_xlim(0, nstep)
        ax[1].set_ylim(-1.05, 1.05)
        plt.tight_layout()
        camera.snap()

    animation = camera.animate()
    animation.save(save_path + '/cl_animation_train.gif', writer='imagemagick')


class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()


class VisualizerOpen(Visualizer):

    def __init__(self, model, verbosity, savedir, training_visuals=False, trace_movie=False, figname=None):
        """

        :param model:
        :param verbosity:
        :param savedir:
        :param training_visuals:
        """
        self.model = model
        self.verbosity = verbosity
        if training_visuals:
            self.anime = Animator(model)
        self.training_visuals = training_visuals
        self.savedir = savedir
        self.figname = figname
        self.trace_movie = trace_movie

    def train_plot(self, outputs, epoch):
        """

        :param outputs:
        :param epoch:
        :return:
        """
        if self.training_visuals:
            if epoch % self.verbosity == 0:
                self.anime()

    def plot_matrix(self):
        if hasattr(self.model, 'fx'):
            if hasattr(self.model.fx, 'effective_W'):
                rows = 1
                mat = self.model.fx.effective_W().detach().cpu().numpy()
            elif isinstance(self.model.fx, blocks.Linear):
                rows = 1
                mat = self.model.fx.linear.effective_W().detach().cpu().numpy()
            elif hasattr(self.model.fx, 'linear'):
                rows = len(self.model.fx.linear)
                Mat = []
                for linear in self.model.fx.linear:
                    Mat.append(linear.weight.detach().cpu().numpy())
            else:
                rows = 0
        elif hasattr(self.model, 'fxud'):
            if hasattr(self.model.fxud, 'effective_W'):
                rows = 1
                mat = self.model.fxud.effective_W().detach().cpu().numpy()
            elif hasattr(self.model.fxud, 'linear'):
                rows = len(self.model.fxud.linear)
                Mat = []
                for linear in self.model.fxud.linear:
                    Mat.append(linear.weight.detach().cpu().numpy())
            else:
                rows = 0
        else:
            rows = 0

        plt.style.use('dark_background')
        if rows == 1:
            fig, (eigax, matax) = plt.subplots(rows, 2)
            eigax.set_aspect(1)
            matax.axis('off')
            matax.set_title('State Transition Weights')
            matax.imshow(mat.T)
            if not mat.shape[0] == mat.shape[1]:
                # singular values of rectangular matrix
                s, w, d = np.linalg.svd(mat.T)
                eigax.set_title('Weights Singular values')
            else:
                w, v = LA.eig(mat.T)
                eigax.set_title('Weights Eigenvalues')
            eigax.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
            patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
            eigax.add_patch(patch)
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))
        elif rows > 1:
            fig, axes = plt.subplots(rows, 2)
            # axes[0, 0].set_title('Weights Eigenvalues')
            axes[0, 1].set_title('State Transition Weights')
            count = 0
            for k in range(rows):
                # axes[k, 0].set_ylim(-1.1, 1.1)
                # axes[k, 0].set_xlim(-1.1, 1.1)
                axes[k, 0].set_aspect(1)
                axes[k, 1].axis('off')
                axes[k, 1].imshow(Mat[k].T)
                if not Mat[k].shape[0] == Mat[k].shape[1]:
                    # singular values of rectangular matrix
                    s, w, d = np.linalg.svd(Mat[k].T)
                    axes[k, 0].set_title('Weights Singular values')
                else:
                    w, v = LA.eig(Mat[k].T)
                    if count == 0:
                        axes[k, 0].set_title('Weights Eigenvalues')
                    count += 1
                axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                patch = mpatches.Circle((0, 0), radius=1, alpha=0.2)
                axes[k, 0].add_patch(patch)
            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir, 'eigmat.png'))

    def plot_traj(self, true_traj, pred_traj, figname='open_loop.png'):
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(len(true_traj), 1)
            labels = [f'$y_{k}$' for k in range(len(true_traj))]
            for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
                axe = ax if len(true_traj) == 1 else ax[row]
                axe.set_ylabel(label, rotation=0, labelpad=20)
                axe.plot(t1, label='True', c='c')
                axe.plot(t2, label='Pred', c='m')
                axe.tick_params(labelbottom=False)
            axe.tick_params(labelbottom=True)
            axe.set_xlabel('Time')
            axe.legend()
        except:
            pass

        plt.savefig(figname)

    def train_output(self):
        """

        :return:
        """
        if self.training_visuals:
            try:
                self.anime.make_and_save(os.path.join(self.savedir, 'eigen_animation.mp4'))
            except ValueError:
                pass
        return dict()

    def eval(self, outputs):
        """

        :param outputs:
        :return:
        """
        dsets = ['train', 'dev', 'test']
        ny = outputs["nstep_train_Y_pred_dynamics"].shape[-1]
        # Ypred = [unbatch_tensor(outputs[f'nstep_{dset}_Y_pred_dynamics']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        # Ytrue = [unbatch_tensor(outputs[f'nstep_{dset}_Yf']).reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ypred = [outputs[f'nstep_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'nstep_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]

        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'nstep_loop.png'))

        figname = self.figname if self.figname is not None else os.path.join(self.savedir, 'open_loop.png')
        Ypred = [outputs[f'loop_{dset}_Y_pred_dynamics'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       figname=figname)
        self.plot_matrix()

        if self.trace_movie:
            trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                                  np.concatenate(Ypred).transpose(1, 0),
                                  figname=os.path.join(self.savedir, f'open_movie.mp4'),
                                  freq=self.verbosity)
        return dict()


class VisualizerUncertaintyOpen(Visualizer):
    def __init__(self, dataset, savedir, dynamics_name='dynamics'):
        self.dataset = dataset
        self.savedir = savedir
        self.dynamics_name = dynamics_name

    def plot_traj(self, true_traj, pred_traj, pred_mean, pred_std, figname='open_loop.png'):
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(len(true_traj), 1)
            labels = [f'$y_{k}$' for k in range(len(true_traj))]
            for row, (t1, t2, mean, std, label) in enumerate(zip(true_traj, pred_traj, pred_mean, pred_std, labels)):
                axe = ax if len(true_traj) == 1 else ax[row]
                x = np.array(range(len(mean)))
                axe.set_ylabel(label, rotation=0, labelpad=20)
                axe.plot(t1, label='True', c='c')
                axe.fill_between(x, mean - std, mean + std, color='C3', alpha=0.2, linewidth=0)
                axe.fill_between(x, mean - std * 2, mean + std * 2, color='C3', alpha=0.2, linewidth=0)
                axe.plot(mean, label='Mean', c='C3')
                axe.tick_params(labelbottom=False)
            axe.tick_params(labelbottom=True)
            axe.set_xlabel('Time')
            axe.legend()
        except Exception as e:
            print(e)

        plt.savefig(figname)

    def eval(self, outputs):
        """
        :param outputs:
        :return:
        """
        dsets = ['train', 'dev', 'test']
        ny = self.dataset.dims['Yf'][-1]

        Ypred = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}'].reshape(-1, ny).detach().cpu().numpy() for dset in
                 dsets]
        Ymean = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}_mean'].reshape(-1, ny).detach().cpu().numpy() for
                 dset in dsets]
        Ystd = [outputs[f'loop_{dset}_Y_pred_{self.dynamics_name}_std'].reshape(-1, ny).detach().cpu().numpy() for dset
                in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].reshape(-1, ny).detach().cpu().numpy() for dset in dsets]
        self.plot_traj(np.concatenate(Ytrue).transpose(1, 0),
                       np.concatenate(Ypred).transpose(1, 0),
                       np.concatenate(Ymean).transpose(1, 0),
                       np.concatenate(Ystd).transpose(1, 0),
                       figname=os.path.join(self.savedir, 'open_loop.png'))
        return dict()


class VisualizerTrajectories(Visualizer):

    def __init__(self, model, plot_keys, verbosity):
        self.model = model
        self.verbosity = verbosity
        self.plot_keys = plot_keys

    def eval(self, outputs):
        data = {k:  unbatch_tensor(v).squeeze(1).detach().cpu().numpy()
                for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        for k, v in data.items():
            plot_traj({k: v}, figname=None)
        return dict()


class VisualizerClosedLoop(Visualizer):
    def __init__(self, u_key='U_pred_policy', y_key='Y_pred_dynamics',
                 r_key='Rf', d_key=None, ymin_key=None, ymax_key=None,
                 umin_key=None, umax_key=None, policy=None,
                 ctrl_outputs=None, savedir='test_control'):
        """

        :param u_key:
        :param y_key:
        :param r_key:
        :param d_key:
        :param ymin_key:
        :param ymax_key:
        :param umin_key:
        :param umax_key:
        :param policy:
        :param ctrl_outputs:
        :param savedir:
        """
        self.model = policy
        self.u_key = u_key
        self.y_key = y_key
        self.r_key = r_key
        self.d_key = d_key
        self.ymin_key = ymin_key
        self.ymax_key = ymax_key
        self.umin_key = umin_key
        self.umax_key = umax_key
        self.ctrl_outputs = ctrl_outputs
        self.savedir = savedir

    def plot_matrix(self):
        if self.model is not None:
            if hasattr(self.model, 'net'):
                if hasattr(self.model.net, 'effective_W'):
                    rows = 1
                    mat = self.model.net.effective_W().detach().cpu().numpy()
                elif hasattr(self.model.net, 'linear'):
                    rows = len(self.model.net.linear)
                    Mat = []
                    for linear in self.model.net.linear:
                        Mat.append(linear.weight.detach().cpu().numpy())
                else:
                    rows = 0
            # plt.style.use('dark_background')
            if rows == 1:
                fig, (eigax, matax) = plt.subplots(rows, 2)
                # eigax.set_ylim(-1.1, 1.1)
                # eigax.set_xlim(-1.1, 1.1)
                eigax.set_aspect(1)
                matax.axis('off')
                matax.set_title('Policy Weights')
                matax.imshow(mat.T)
                if not mat.shape[0] == mat.shape[1]:
                    # singular values of rectangular matrix
                    s, w, d = np.linalg.svd(mat.T)
                    eigax.set_title('Weights Singular values')
                else:
                    w, v = LA.eig(mat.T)
                    eigax.set_title('Weights Eigenvalues')
                eigax.scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                plt.tight_layout()
                plt.savefig(os.path.join(self.savedir, 'eigmat.png'))
            elif rows > 1:
                fig, axes = plt.subplots(rows, 2)
                # axes[0, 0].set_title('Weights Eigenvalues')
                axes[0, 1].set_title('State Transition Weights')
                count = 0
                for k in range(rows):
                    axes[k, 0].set_aspect(1)
                    axes[k, 1].axis('off')
                    axes[k, 1].imshow(Mat[k].T)
                    if not Mat[k].shape[0] == Mat[k].shape[1]:
                        # singular values of rectangular matrix
                        s, w, d = np.linalg.svd(Mat[k].T)
                        axes[k, 0].set_title('Weights Singular values')
                    else:
                        w, v = LA.eig(Mat[k].T)
                        axes[k, 0].set_title('Weights Eigenvalues') if count == 0 else None
                        count += 1
                    axes[k, 0].scatter(w.real, w.imag, alpha=0.5, c=get_colors(len(w.real)))
                plt.tight_layout()
                plt.savefig(os.path.join(self.savedir, 'eigmat.png'))

    def eval(self, outputs, plot_weights=False, figname='CL_control.png'):

        Y = outputs[self.y_key].detach().numpy() if self.y_key is not None else None
        U = outputs[self.u_key].detach().numpy() if self.u_key is not None else None
        D = outputs[self.d_key].detach().numpy() if self.d_key is not None else None
        R = outputs[self.r_key].detach().numpy() if self.r_key is not None else None
        Ymin = outputs[self.ymin_key].detach().numpy() if self.ymin_key is not None else None
        Ymax = outputs[self.ymax_key].detach().numpy() if self.ymax_key is not None else None
        Umin = outputs[self.umin_key].detach().numpy() if self.umin_key is not None else None
        Umax = outputs[self.umax_key].detach().numpy() if self.umax_key is not None else None
        ctrl_outputs = None if self.ctrl_outputs is None else self.ctrl_outputs

        pltCL(Y=Y, U=U, D=D, R=R,
                   Ymin=Ymin, Ymax=Ymax, Umin=Umin, Umax=Umax,
                   ctrl_outputs=ctrl_outputs,
                   figname=os.path.join(self.savedir, 'CL_control.png'))
        if plot_weights:
            self.plot_matrix()
        return dict()


class VisualizerDobleIntegrator(Visualizer):
    """
    custom visualizer for double integrator example
    """
    def __init__(self, dataset, model, policy, dynamics, verbosity, savedir,
                 nstep=40, x0=1.5 * np.ones([2, 1]),
                 training_visuals=False, trace_movie=False):
        """

        :param dataset:
        :param model:
        :param verbosity:
        :param savedir:
        :param training_visuals:
        :param trace_movie:
        """
        self.model = model
        self.policy = policy
        self.dynamics = dynamics
        self.dataset = dataset
        self.verbosity = verbosity
        self.savedir = savedir
        self.training_visuals = training_visuals
        self.trace_movie = trace_movie
        self.nstep = nstep
        self.x0 = x0

    def train_output(self, trainer, epoch_policy):
        """
        visualize evolution of closed-loop contro and policy landscape during training
        :return:
        """
        A = self.dynamics.fx.linear.weight
        B = self.dynamics.fu.linear.weight
        if self.training_visuals:
            X_list, U_list = [], []
            policy_list = []
            for i in range(trainer.epochs):
                best_policy = epoch_policy[i]
                trainer.model.eval()
                policy = self.policy
                policy.load_state_dict(best_policy)
                X, U = cl_simulate(A, B, policy.net, nstep=self.nstep, x0=self.x0)
                X_list.append(X)
                U_list.append(U)
                policy_list.append(best_policy)
            plot_cl_train(X_list, U_list, nstep=self.nstep, save_path=self.savedir)
            plot_policy_train(A, B, policy, policy_list, save_path=self.savedir)
        return dict()

    def eval(self, trainer):

        A = self.dynamics.fx.linear.weight
        B = self.dynamics.fu.linear.weight
        policy = self.policy
        # plot closed loop trajectories
        X, U = cl_simulate(A, B, policy.net, nstep=self.nstep, x0=self.x0)
        plot_cl(X, U, nstep=self.nstep, save_path=self.savedir, trace_movie=self.trace_movie)
        # plot policy surface
        plot_policy(policy.net, save_path=self.savedir)
        # loss landscape and contraction regions
        plot_loss_DPC(trainer.model, policy, A, B, trainer.train_data, xmin=-2, xmax=2,
                  save_path=self.savedir)
        return dict()
