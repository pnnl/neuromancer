"""
Two area problem

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import slim
import psl

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Objective, Loss
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import read_file, normalize_data, split_static_data, StaticDataset
from neuromancer.component import Function
import neuromancer.blocks as blocks


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
           help="loss function weight.")
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=1.0,
           help="constraints penalty weight.")
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=3000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.05,
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
    # # #  Optimization problem hyperparameters
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
    data = read_file(psl.datasets[system])
    # TODO: update datasets to load multi-experiment static dataset from multiple files
    nsim = data['Y'].shape[0]
    # dataset variables
    V = data['Y'][:,[0]]
    theta_V = data['Y'][:,[1]]
    I = data['Y'][:,[2]]
    theta_I = data['Y'][:,[3]]
    P = V*I*np.cos(theta_V-theta_I)
    Q = V*I*np.sin(theta_V-theta_I)
    # Constants - first time instance of P*(t), Q*(t), V(t)
    P_0 = P[0, 0]
    Q_0 = Q[0, 0]
    V_0 = V[0, 0]
    # data loaders for training
    samples = {"V(t)": V, "I(t)": I, "theta_V": theta_V, "theta_I": theta_I, "P*(t)": P, "Q*(t)": Q}
    # norm_type = None
    norm_type = "one-one"
    # norm_type = "zscore"
    split_data, dims = get_dataloaders(samples, norm_type=norm_type)
    train_data, dev_data, test_data = split_data

    """
    # # #  Constrained regression problem formulation in Neuromancer
    """
    n_var = 2           # number of decision variables: P(t), Q(t)
    # define solution map as MLP
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
        # input_keys=["V(t)", "I(t)", "theta_V", "theta_I"],
        name='regressor',
    )

    # TODO: use of function under construction
    n_out = 2           # number of output variables: P(t), Q(t)
    n_in = 1           # number of input variables: V(t)
    # define neural net
    net = blocks.MLP(insize=n_in,
                     outsize=n_out,
                     bias=args.bias,
                     linear_map=linmap,
                     nonlin=activation,
                     hsizes=[args.nx_hidden] * args.n_layers,)
    # define neuromancer function mapping input keys to output keys via function (nn.Module)
    nn_map = Function(func=net, input_keys=["V(t)"], output_keys=["PQ(t)"],
                      name='net')

    # dataset variables
    V_t = Variable('V(t)')
    P_star = Variable('P*(t)')
    Q_star = Variable('Q*(t)')
    # neural model output variables
    P_hat = Variable(f"U_pred_{sol_map.name}")[:, 0]
    Q_hat = Variable(f"U_pred_{sol_map.name}")[:, 1]
    # P_hat = Variable(nn_map.output_keys)[:, 0]
    # Q_hat = Variable(nn_map.output_keys)[:, 1]

    # trainable parameters of the polynomial regressor as variables
    alpha_p = Variable('alpha_p', value=torch.nn.Parameter(torch.tensor(0.1)))
    alpha_i = Variable('alpha_i', value=torch.nn.Parameter(torch.tensor(0.1)))
    alpha_z = Variable('alpha_z', value=torch.nn.Parameter(torch.tensor(0.1)))
    beta_p = Variable('beta_p', value=torch.nn.Parameter(torch.tensor(0.1)))
    beta_i = Variable('beta_i', value=torch.nn.Parameter(torch.tensor(0.1)))
    beta_z = Variable('beta_z', value=torch.nn.Parameter(torch.tensor(0.1)))
    # polynomial model output variables
    P_tilde = P_0*(alpha_p+alpha_i*V_t/V_0+alpha_z*V_t**2/V_0**2)
    Q_tilde = Q_0*(beta_p+beta_i*V_t/V_0+beta_z*V_t**2/V_0**2)

    # convex combination of polynomial and neural model
    a, b = 0.5, 0.5         # a,b = 1 -> polynomial model,  a,b = 0 -> neural model
    P_mix = a*P_tilde + (1-a)*P_hat
    Q_mix = b*Q_tilde + (1-b)*Q_hat

    # objective function
    loss_1 = args.Q*(P_mix - P_star == 0)^2
    loss_1.name = 'P_loss'
    loss_2 = args.Q*(Q_mix - Q_star == 0)^2
    loss_2.name = 'Q_loss'

    # loss_1 = args.Q*(P_hat - P_star == 0)^2
    # loss_1.name = 'P_loss'
    # loss_2 = args.Q*(Q_hat - Q_star == 0)^2
    # loss_2.name = 'Q_loss'

    # loss_1 = args.Q*(P_tilde - P_star == 0)^2
    # loss_1.name = 'P_loss'
    # loss_2 = args.Q*(Q_tilde - Q_star == 0)^2
    # loss_2.name = 'Q_loss'

    # constraints
    con_1 = args.Q_con*(alpha_p >= 0)
    con_2 = args.Q_con*(alpha_i >= 0)
    con_3 = args.Q_con*(alpha_z >= 0)
    con_4 = args.Q_con*(beta_p >= 0)
    con_5 = args.Q_con*(beta_i >= 0)
    con_6 = args.Q_con*(beta_z >= 0)

    # constrained optimization problem construction
    objectives = [loss_1, loss_2]
    # objectives = [loss_1]
    components = [sol_map]
    # components = [nn_map]
    # constraints = []
    constraints = [con_1, con_2, con_3, con_4, con_5, con_6]
    model = Problem(objectives, constraints, components)
    model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_two_area'
    args.verbosity = 1
    metrics = ["train_loss", "dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'two_area'

    """
    # # #  Constrained least squares problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
        eval_metric="train_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # learn solution
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)


    """
    # # #  Plots
    """
    # get the dict of the whole dataset
    if norm_type is not None:
        samples, _ = normalize_data(samples, norm_type=norm_type)
    all_data = StaticDataset(samples, name="dataset")
    all_data_dict = all_data.get_full_batch()
    # normalized targets
    P = all_data_dict['P*(t)']
    Q = all_data_dict['Q*(t)']
    # normalized independent variables
    V = all_data_dict['V(t)']
    I = all_data_dict['I(t)']
    theta_V = all_data_dict['theta_V']
    theta_I = all_data_dict['theta_I']
    # outputs of the trained polynomial model
    P_tilde_out = P_tilde(all_data_dict).detach().numpy()
    Q_tilde_out = Q_tilde(all_data_dict).detach().numpy()
    # outputs of the trained neural model
    net_out = sol_map(all_data_dict)
    P_hat_out = net_out[f"U_pred_{sol_map.name}"][0, :, [0]].detach().numpy()
    Q_hat_out = net_out[f"U_pred_{sol_map.name}"][0, :, [1]].detach().numpy()
    # combined model
    P_mix_out = a*P_tilde_out + (1-a)*P_hat_out
    Q_mix_out = b*Q_tilde_out + (1-b)*Q_hat_out

    # plot regressors
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(P)
    ax[0].plot(P_mix_out)
    ax[0].plot(P_tilde_out)
    ax[0].plot(P_hat_out)
    ax[0].set(ylabel='$P$')
    ax[0].legend(['P', 'P_mix', 'P_tilde', 'P_hat'])
    ax[1].plot(Q)
    ax[1].plot(Q_mix_out)
    ax[1].plot(Q_tilde_out)
    ax[1].plot(Q_hat_out)
    ax[1].legend(['Q', 'Q_mix', 'Q_tilde', 'Q_hat'])
    ax[1].set(ylabel='$Q$')
    ax[1].set(xlabel='$time$')
    # plot independent variables
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(V)
    ax[0].set(ylabel='$V$')
    ax[1].plot(I)
    ax[1].set(ylabel='I')
    ax[2].plot(theta_V)
    ax[2].set(ylabel='$theta_V$')
    ax[3].plot(theta_I)
    ax[3].set(ylabel='$theta_I$')
    ax[3].set(xlabel='$time$')


    ###########################################
    # TODO: testing function
    train_d = train_data.dataset.get_full_batch()
    net_out = sol_map(train_d)
    fun_out = nn_map(train_d)

    x = [train_d[k] for k in nn_map.input_keys]
    out = nn_map.func(*x)
    fun_out2 = {
            k: v for k, v in zip(
                nn_map.output_keys,
                out if isinstance(out, tuple) else (out,)
            )
        }
