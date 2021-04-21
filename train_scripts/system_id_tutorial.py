
import torch
import torch.nn.functional as F
import slim
import psl

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.simulators import OpenLoopSimulator
from neuromancer.datasets import load_dataset
from neuromancer.loggers import BasicLogger


if __name__ == "__main__":

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  DATASET LOAD   # # # # # # # #
    # # # # # # # # # # # # # # # # # # #
    """

    # for available systems in PSL library check: psl.systems.keys()
    # for available datasets in PSL library check: psl.datasets.keys()
    system = 'new_system'         # keyword of selected system

    # load argument parser
    parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(system=system),
                                    arg.loss(), arg.lin(), arg.ssm()])
    grp = parser.group('OPTIMIZATION')
    grp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    args, grps = parser.parse_arg_groups()
    device = "cpu"
    # metrics to be logged
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    dataset = load_dataset(args, device, 'openloop')
    print(dataset.dims)
    # nsim = number of time steps in the dataset time series
    # nsteps = legth of the prediction horizon
    # Y = observed outputs
    # U = inputs
    # D = disturbances
    # Yp = past trajectories generated as Y[0:-nsteps]
    # Xf = future trajectories generated as Y[nesteps:]

    #  Train, Development, Test sets
    dataset.train_data['Yp'].shape
    dataset.dev_data['Yp'].shape
    dataset.test_data['Yp'].shape
    # out: torch.Size([batch size (prediction horizon),
    #                  number of batches, number of variables])

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  MODEL CONSTRUCTION   # # # # #
    # # # # # # # # # # # # # # # # # # #
    
    # Model = directed acyclic graph of neural components
    # neural components = structured neural networks composed of blocks
    # blocks = standard neural architectures such as MLP, ResNet, RNN,
    #         composed of linear layers and activations
    # linear layer = possibly using constrained/factorized matrices from slim
    # activations = possibly using learnable activation functions
    """

    # code here


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #  POBLEM DEFINITION    # # # # #
    # # # # # # # # # # # # # # # # # # #

    # Problem = model components + constraints + objectives
    """

    # code here


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #     OPTIMIZATION      # # # # #
    # # # # # # # # # # # # # # # # # # #

    #  trainer = problem + dataset + optimizer
    """

    # code here


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #        RESULTS        # # # # #
    # # # # # # # # # # # # # # # # # # #

    # visualize and log results
    """

    # code here