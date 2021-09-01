"""
Two area problem

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Objective, Loss
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import normalize_data, split_static_data, StaticDataset
from neuromancer.plot import plot_loss_mpp, plot_solution_mpp


def arg_mpLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=25.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-nx_hidden", type=int, default=40,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=800,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=200,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=200,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def get_dataloaders(data, norm_type=None, split_ratio=None, num_workers=0):
    """This will generate dataloaders for a given dictionary of data.
    Dataloaders are hard-coded for full-batch training to match NeuroMANCER's training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_static_data(data, split_ratio)

    train_data = StaticDataset(
        train_data,
        name="train",
    )
    dev_data = StaticDataset(
        dev_data,
        name="dev",
    )
    test_data = StaticDataset(
        test_data,
        name="test",
    )

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

    return (train_data, dev_data, test_data), train_data.dataset.dims


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset 
    """


    system = '200bus_boundary'         # keyword of selected system


    np.random.seed(args.data_seed)
    nsim = 10000  # number of datapoints: increase sample density for more robust results
    # TODO: load true train data V(t), P*(t), Q*(t),
    samples = {"V(t)": np.random.uniform(low=0.0, high=5.0, size=(nsim, 1)),
               "P*(t)": np.random.uniform(low=0.0, high=5.0, size=(nsim, 1)),
               "Q*(t)": np.random.uniform(low=0.0, high=5.0, size=(nsim, 1))}
    nstep_data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  constrained regression problem formulation in Neuromancer
    """
    n_var = 2           # number of decision variables: P(t), Q(t)
    # define solution map as MLP policy
    dims['U'] = (nsim, n_var)  # defining expected dimensions of the solution variable: internal policy key 'U'
    activation = activations['relu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["V(t)"],
        name='regressor',
    )

    # variables
    V_t = Variable('V(t)')
    P_star = Variable('P*(t)')
    Q_star = Variable('Q*(t)')
    P_net = Variable(f"U_pred_{sol_map.name}")[:, 0]
    Q_net = Variable(f"U_pred_{sol_map.name}")[:, 1]

    # Constants
    P_0 = 1
    Q_0 = 1
    V_0 = 1

    # alpha_p = torch.relu(torch.nn.Parameter(torch.tensor(0.1)))

    # TODO: support variable initialized with parameter values to log parameters in problem params
    # trainable parameters of the polynomial regressor as variables
    alpha_p = Variable('alpha_p', value= torch.nn.Parameter(torch.tensor(0.1)))
    alpha_i = Variable('alpha_i', value= torch.nn.Parameter(torch.tensor(0.1)))
    alpha_z = Variable('alpha_z', value= torch.nn.Parameter(torch.tensor(0.1)))
    beta_p = Variable('beta_p', value= torch.nn.Parameter(torch.tensor(0.1)))
    beta_i = Variable('beta_i', value= torch.nn.Parameter(torch.tensor(0.1)))
    beta_z = Variable('beta_z', value= torch.nn.Parameter(torch.tensor(0.1)))
    a = Variable('a', value= torch.nn.Parameter(torch.tensor(0.1)))
    b = Variable('b', value= torch.nn.Parameter(torch.tensor(0.1)))

    # polynomial model
    P = P_0*(alpha_p+alpha_i*V_t/V_0+alpha_z*V_t**2/V_0)
    Q = Q_0*(beta_p+beta_i*V_t/V_0+beta_z*V_t**2/V_0)

    # objective function
    # loss_1 = args.Q*(P - P_star == 0)^2
    # loss_1.name = 'P_loss'
    # loss_2 = args.Q*(Q - Q_star == 0)^2
    # loss_2.name = 'Q_loss'

    loss_1 = args.Q*(P_net - P_star == 0)^2
    loss_1.name = 'P_loss'
    loss_2 = args.Q*(Q_net - Q_star == 0)^2
    loss_2.name = 'Q_loss'

    # constraints
    con_1 = args.Q_con*(alpha_p + alpha_i + alpha_z == a)
    con_2 = args.Q_con*(beta_p + beta_i + beta_z == b)

    # TODO: we can avoid these soft constraints by using projections via ReLUs in the model architecture
    con_3 = args.Q_con*(alpha_p >= 0)
    con_4 = args.Q_con*(alpha_i >= 0)
    con_5 = args.Q_con*(alpha_z >= 0)
    con_6 = args.Q_con*(beta_p >= 0)
    con_7 = args.Q_con*(beta_i >= 0)
    con_8 = args.Q_con*(beta_z >= 0)
    con_9 = args.Q_con*(a >= 0)
    con_10 = args.Q_con*(b >= 0)

    # constrained optimization problem construction
    objectives = [loss_1, loss_2]
    constraints = []
    components = [sol_map]
    # constraints = [con_1, con_2, con_3, con_4, con_5, con_6, con_7, con_8, con_9, con_10]
    # components = []
    model = Problem(objectives, constraints, components)
    model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_two_area'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'two_area'

    """
    # # #  mpQP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(param)

    # define trainer
    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

