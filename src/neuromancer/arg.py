"""
This module contains an extension of the argparse.ArgumentParser class and some
parsers that are generally useful for writing training scripts using the neuromancer library.
ArgumentParser is extended to take advantage of grouped arguments when passing
command line arguments to functions and to abbreviate argparse's verbose method names.
"""

import argparse
from argparse import ArgumentParser

from neuromancer.modules.activations import activations


def add(self, argname, **kwargs):
    """
    Monkey patch for argument group objects.

    :param self: Refers to instantiated object of class being patched
    :param argname: (str) Command line argument
    :param kwargs: Whatever keyword args this thing accepts
    :return: Whatever add_argument returns (I think it is an Action object)
    """
    if argname.startswith('--'):
        argname = argname[:2], argname[2:]
    elif argname.startswith('-'):
        argname = argname[:1], argname[1:]
    else:
        argname = '', argname
    return self.add_argument(f'{argname[0]}{self.prefix}{argname[1]}', **kwargs)


argparse.ArgumentParser.add = add
argparse._ArgumentGroup.add = add


class ArgParser(ArgumentParser):
    """
    Subclass argparser for abbreviated method calls, separate namespaces for argument groups, and optional command line
    prefix so that we can reuse parser definitions. 
    """
    def __init__(self, prefix='', **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def check_for_group(self, group_name):
        """
        This function will return the argument group if it exists

        :param group_name: (str) Name of the argument group
        :return: (argparse._ArgumentGroup or None)
        """
        for gp in self._action_groups:
            if gp.title == group_name:
                return gp
        return None

    def group(self, group_name, prefix=None):
        """
        Monkey patch to abbreviate verbose call to add_argument_group. If an argument group exists by
        the name group_name it will be returned, otherwise a new argument group will be created and returned.
        
        :param group_name: (str) Name of the argument group
        :return: argparse._ArgumentGroup
        """
        gp = self.check_for_group(group_name)
        if gp is None:
            gp = self.add_argument_group(group_name)
        if prefix is None:
            gp.prefix = self.prefix
        return gp

    def parse_arg_groups(self):
        """

        :return: (Namespace, List(Namespace))
        """
        args = self.parse_args()
        print(args)
        arg_groups = {}
    
        for group in self._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_groups[group.title] = argparse.Namespace(**group_dict)
        return args, arg_groups


def log(prefix=''):
    """
    Command line parser for logging arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
    are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    parser.add("-gpu", type=int, help="GPU to use")
    parser.add("-seed", type=int, default=408, help="Random seed used for weight initialization.")
    gp = parser.group("LOGGING")

    gp.add("-savedir", type=str, default="test",
           help="Where should your trained model and plots be saved (temp)")

    gp.add("-verbosity", type=int, default=1,
           help="How many epochs in between status updates")

    gp.add("-exp", type=str, default="test",
           help="Will group all run under this experiment name.")

    gp.add("-location", type=str, default="mlruns",
           help="Where to write mlflow experiment tracking stuff")

    gp.add("-run", type=str, default="neuromancer",
           help="Some name to tell what the experiment run was about.")

    gp.add("-logger", type=str, choices=["mlflow", "stdout"], default="stdout",
           help="Logging setup to use")

    return parser


def opt(prefix=''):
    """
    Command line parser for optimization arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("OPTIMIZATION")

    gp.add("-epochs", type=int, default=20,
           help='Number of training epochs')

    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")

    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")

    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")

    gp.add("-skip_eval_sim", action="store_true",
           help="Whether to run simulator during evaluation phase of training.")

    return parser


def data(prefix='', system='CSTR'):
    """
    Command line parser for data arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("DATA")

    gp.add("-nsteps", type=int, default=32,
           help="Number of steps for open loop during training.")

    gp.add("-dataset", type=str, default=system,
           help="select particular dataset with keyword")

    gp.add("-nsim", type=int, default=10000,
           help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
                "train, dev, and test will be split evenly from contiguous, sequential, "
                "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
                "next nsim/3 are dev and next nsim/3 simulation steps are test points."
                "None will use a default nsim from the selected dataset or emulator")

    gp.add("-norm", nargs="+", default=["U", "D", "Y"], choices=["U", "D", "Y", "X"],
           help="List of sequences to max-min normalize")

    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")

    return parser


def lin(prefix=''):
    """
    Command line parser for linear map arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("LINEAR")
    
    gp.add("-linear_map", type=str, choices=["linear", "softSVD", "pf"], default="linear",
           help='Choice of map from SLiM package')

    gp.add("-sigma_min", type=float, default=0.1,
           help='Minimum singular value (for maps with singular value constraints)')

    gp.add("-sigma_max", type=float, default=1.0,
           help='Maximum singular value (for maps with singular value constraints)')

    return parser


def loss(prefix=''):
    """
    Command line parser for loss function arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("LOSS")
    
    gp.add("-Q_con_x", type=float, default=1.0,
           help="Hidden state constraints penalty weight.")

    gp.add("-Q_dx", type=float, default=0.1,
           help="Penalty weight on hidden state difference in one time step.")

    gp.add("-Q_sub", type=float, default=0.1,
           help="Linear maps regularization weight.")

    gp.add("-Q_y", type=float, default=1.0,
           help="Output tracking penalty weight")

    gp.add("-Q_e", type=float, default=1.0,
           help="State estimator hidden prediction penalty weight")

    gp.add("-Q_con_fdu", type=float, default=0.0,
           help="Penalty weight on control actions and disturbances.")

    return parser


def freeze(prefix=''):
    """
    Command line parser for weight freezing arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group('FREEZE')

    gp.add("-freeze", nargs="+", default=[""],
           help="sets requires grad to False")

    gp.add("-unfreeze", default=["components.2"],
           help="sets requires grad to True")

    return parser


def ctrl_loss(prefix=''):
    """
    Command line parser for special control loss arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)

    gp = parser.group('LOSS')
    gp.add("-Q_con_u", type=float, default=0.0,
           help="Input constraints penalty weight.")

    gp.add("-Q_r", type=float, default=1.0,
           help="Reference tracking penalty weight")

    gp.add("-Q_du", type=float, default=0.0,
           help="control action difference penalty weight")

    gp.add("-con_tighten", action="store_true",
           help='Tighten constraints')

    gp.add("-tighten", type=float, default=0.00,
           help="control action difference penalty weight")

    gp.add("-loss_clip", action="store_true",
           help='Clip loss terms to avoid terms taking over at beginning of training')

    gp.add("-noise", action="store_true",
           help='Whether to add noise to control actions during training.')

    gp.add("-Q_con_y", type=float, default=0.0,
           help="Observable constraints penalty weight.")

    return parser


def ssm(prefix=''):
    """
    Command line parser for state space model arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group('SSM')

    gp.add("-ssm_type", type=str, choices=["blackbox", "hw", "hammerstein", "blocknlin", "linear"], default="hammerstein",
           help='Choice of block structure for system identification model')

    gp.add("-nx_hidden", type=int, default=32,
           help="Number of hidden states per output")

    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of single time-step state transition")

    gp.add("-state_estimator", type=str, choices=["rnn", "mlp", "linear", "residual_mlp", "fully_observable"], default="mlp",
           help='Choice of model architecture for state estimator.')

    gp.add("-estimator_input_window", type=int, default=8,
           help="Number of previous time steps measurements to include in state estimator input")

    gp.add("-nonlinear_map", type=str, default="mlp", choices=["mlp", "rnn", "pytorch_rnn", "linear", "residual_mlp"],
           help='Choice of architecture for component blocks in state space model.')

    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")

    gp.add("-activation", choices=activations.keys(), default="gelu",
           help="Activation function for component block neural networks")

    return parser


def policy(prefix=''):
    """
    Command line parser for control policy arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("POLICY")

    gp.add("-policy", type=str, choices=["mlp", "linear"], default="mlp",
           help='Choice of architecture for modeling control policy.')

    gp.add("-controlled_outputs", type=int, nargs='+', default=[0],
           help="list of indices of controlled outputs len(default)<=ny")

    gp.add("-n_hidden", type=int, default=20,
           help="Number of hidden states")

    gp.add("-n_layers", type=int, default=3,
           help="Number of hidden layers of single time-step state transition")

    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network models.")

    gp.add("-policy_features", nargs="+", default=['Y_ctrl_p', 'Rf', 'Y_maxf', 'Y_minf'],
           help="Policy features")  # reference tracking option

    gp.add("-activation", choices=["gelu", "softexp"], default="gelu",
           help="Activation function for neural networks")

    gp.add("-perturbation", choices=["white_noise_sine_wave", "white_noise"], default="white_noise",
           help='System perturbation method.')

    return parser
