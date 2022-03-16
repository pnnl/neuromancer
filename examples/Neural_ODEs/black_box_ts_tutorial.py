# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg, integrators, ode
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset
from neuromancer.constraint import Variable
from neuromancer.loss import PenaltyLoss, BarrierLoss


def arg_sys_id_problem(prefix=''):
    """
    Command line parser for system identification problem arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("system_id")
    #  DATA
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    #  LOG
    gp.add("-savedir", type=str, default="test",
           help="Where should your trained model and plots be saved (temp)")
    gp.add("-verbosity", type=int, default=1,
           help="How many epochs in between status updates")
    gp.add("-metrics", nargs="+", default=["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"],
           help="Metrics to be logged")
    #  OPTIMIZATION
    gp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    gp.add("-epochs", type=int, default=200,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-skip_eval_sim", action="store_true",
           help="Whether to run simulator during evaluation phase of training.")
    gp.add("-seed", type=int, default=408, help="Random seed used for weight initialization.")
    gp.add("-device", type=str, default="cpu", choices=["cpu", "gpu"],
           help="select device")
    gp.add("-loss", type=str, default='penalty',
           choices=['penalty', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-batch_second", default=True, choices=[True, False],
           help="whether the batch is a second dimension in the dataset.")
    return parser


# %%
def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type=None, split_ratio=None, num_workers=0,
):
    """This will generate dataloaders and open-loop sequence dictionaries for a given dictionary of
    data. Dataloaders are hard-coded for full-batch training to match NeuroMANCER's original
    training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param nsteps: (int) length of windowed subsequences for N-step training.
    :param moving_horizon: (bool) whether to use moving horizon batching.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """
    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_sequence_data(data, nsteps, moving_horizon, split_ratio)

    train_data = SequenceDataset(
        train_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="train",
    )
    dev_data = SequenceDataset(
        dev_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="dev",
    )
    test_data = SequenceDataset(
        test_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="test",
    )

    train_loop = train_data.get_full_sequence()
    dev_loop = dev_data.get_full_sequence()
    test_loop = test_data.get_full_sequence()

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), (train_loop, dev_loop, test_loop), train_data.dataset.dims


def get_loss(objectives, constraints, loss):
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints, batch_second=args.batch_second)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints, barrier=args.barrier_type,
                           batch_second=args.batch_second)
    return loss


if __name__ == "__main__":

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  ARGS and LOGGER  # # # # # # # 
    # # # # # # # # # # # # # # # # # # #
    """
    # load argument parser
    parser = arg.ArgParser(parents=[arg_sys_id_problem()])
    args, grps = parser.parse_arg_groups()

    """
    
    Get some data from the L-V system for prototyping.
    
    """
    # Get the data by simulating system in PSL:
    system = psl.systems['Brusselator1D']

    modelSystem = system()
    ts = 0.05
    raw = modelSystem.simulate(ts=ts)
    psl.plot.pltOL(Y=raw['Y'])
    psl.plot.pltPhase(X=raw['Y'])

    #  Train, Development, Test sets - nstep and loop format
    nsteps = 1
    nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps, moving_horizon=True)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # %% Identity mapping
    nx = 2
    estim = estimators.FullyObservable(
        {**train_data.dataset.dims, "x0": (nx,)},
        linear_map=slim.maps['identity'],
        input_keys=["Yp"],
    )

    # %% Instantiate the blocks, dynamics model:
    fx = blocks.MLP(nx, nx, linear_map=slim.maps['linear'],
                    nonlin=activations['leakyrelu'],
                    hsizes=[10, 10])
    fxRK4 = integrators.RK4(fx, h=ts)
    fy = slim.maps['identity'](nx, nx)

    dynamics_model = dynamics.ODEAuto(fxRK4, fy, name='dynamics',
                                      input_key_map={"x0": f"x0_{estim.name}"})

    # %% Constraints + losses:
    yhat = Variable(f"Y_pred_{dynamics_model.name}")
    y = Variable("Yf")
    x0 = Variable(f"x0_{estim.name}")
    xhat = Variable(f"X_pred_{dynamics_model.name}")

    yFD = (y[1:] - y[:-1])
    yhatFD = (yhat[1:] - yhat[:-1])

    fd_loss = 2.0*((yFD == yhatFD)^2)
    fd_loss.name = 'FD_loss'

    reference_loss = ((yhat == y)^2)
    reference_loss.name = "ref_loss"

    # %%
    objectives = [reference_loss, fd_loss]
    constraints = []
    components = [estim, dynamics_model]
    # create constrained optimization loss
    loss = get_loss(objectives, constraints, args)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()
    problem = problem.to(args.device)

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    logger = BasicLogger(args=args, savedir=args.savedir,
                         verbosity=args.verbosity, stdout=args.metrics)

    simulator = OpenLoopSimulator(
        problem, train_loop, dev_loop, test_loop, eval_sim=True, device=args.device,
    ) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
        problem, train_loop, dev_loop, test_loop, eval_sim=True, device=args.device,
    )
    visualizer = VisualizerOpen(
        dynamics_model,
        1,
        'test',
        training_visuals=False,
        trace_movie=False,
    )
    callback = SysIDCallback(simulator, visualizer)

    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        callback=callback,
        patience=args.patience,
        warmup=args.warmup,
        epochs=args.epochs,
        eval_metric="nstep_dev_"+reference_loss.output_keys[0],
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        logger=logger,
        device=args.device,
    )
    # %%
    best_model = trainer.train()
    # %%
    best_outputs = trainer.test(best_model)
    # %%


    # %%
    y1 = best_outputs['loop_test_Y_pred_dynamics'].detach().numpy()[:,0,:]
    y0 = test_loop['Yp'].detach().numpy()[:,0,:]
    # %% construct hankel matricies for data and pred:

    def hankel(y,m):
        H = np.zeros((m,len(y0[:,1])-m))
        for j in range(len(H[0,:])):
            H[:,j] = y[j:j+m]
        return H

    H1 = hankel(y1[:, 0], 50)
    H0 = hankel(y0[:, 0], 50)

    u0, s0, vh0 = np.linalg.svd(H0, full_matrices=False)
    u1, s1, vh1 = np.linalg.svd(H1, full_matrices=False)

    # %%
    plt.figure()
    x = np.arange(len(s0))
    plt.scatter(x, np.log10(s0), label='data')
    plt.scatter(x, np.log10(s1), label='pred')
    plt.legend()
    plt.show()
    print(s0[0]/s0[-1])  # condition number of the sigma matrix
    print(s1[0]/s1[-1])  # condition number of the sigma matrix

    # %%
    # %% Vector field?
    xx = torch.tensor(np.linspace(0, 5, 100), dtype=torch.float)
    yy = torch.tensor(np.linspace(0, 5, 100), dtype=torch.float)

    grid_x, grid_y = torch.meshgrid(xx, yy)

    XX = 0.0*grid_x
    YY = 0.0*grid_y

    for ii, xc in enumerate(xx):
        for jj, yc in enumerate(yy):
            XX[jj, ii],YY[jj, ii] = fx(torch.tensor([xc, yc]))

    # %%
    t = torch.arange(1000)
    y = np.zeros((1000, 2))
    y0 = torch.tensor([1.0, 0.0])
    for j, t in enumerate(t):
        if j==0:
            y[j, :] = fxRK4(torch.tensor(y0, dtype=torch.float)).detach().numpy()
        else:
            y[j, :] = fxRK4(torch.tensor(y[j-1, :], dtype=torch.float)).detach().numpy()
    plt.plot(y)
    plt.show()


    # %% Compare vector fields?
    """
    1. Subtract extracted and ground truth vector fields w/in the same range. 
    2. Display error as contour on log scale.
    """

    # Ground truth vector field:
    xxt = np.linspace(0, 5, 100)
    yyt = np.linspace(0, 5, 100)

    grid_xt, grid_yt = np.meshgrid(xxt, yyt)

    XXt = 0.0*grid_xt
    YYt = 0.0*grid_yt

    for ii, xc in enumerate(xxt):
        for jj, yc in enumerate(yyt):
            XXt[jj, ii] = 1.0 + yc*xc**2 -3.0*xc - xc
            YYt[jj, ii] = 3.0*xc - yc*xc**2

    plt.figure()
    plt.streamplot(grid_xt, grid_yt, XXt, YYt)
    plt.plot(raw['Y'][:, 0], raw['Y'][:, 1], color='white', linewidth=3)
    plt.show()

    # %%
    plt.contour(grid_xt, grid_yt, np.log10(abs(XX.detach().numpy() - XXt) + 1e-8), levels=20)
    plt.plot(raw['Y'][:, 0], raw['Y'][:, 1], color='white', linewidth=3)
    plt.plot(y[:, 0], y[:, 1], color='red', linewidth=3)

    # %%
    plt.contour(grid_xt, grid_yt, np.log10(abs(YY.detach().numpy() - YYt) + 1e-8), levels=20)
    plt.plot(raw['Y'][:, 0], raw['Y'][:, 1], color='white', linewidth=3)
    plt.plot(y[:, 0], y[:, 1], color='red', linewidth=3)
    # %%
    plt.streamplot(grid_xt, grid_yt, XX.detach().numpy(), YY.detach().numpy())
    plt.plot(raw['Y'][:, 0], raw['Y'][:, 1], color='white', linewidth=3)
    plt.plot(y[:, 0], y[:, 1], color='red', linewidth=3)
    # %%

    # %%
