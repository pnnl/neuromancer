import argparse

from neuromancer.datasets import FileDataset, EmulatorDataset, systems
from neuromancer import loggers

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=int, help="GPU to use")

    # logging parameters
    log_group = parser.add_argument_group("LOGGING PARAMETERS")
    log_group.add_argument(
        "-savedir",
        type=str,
        default="test",
        help="Where should your trained model and plots be saved (temp)",
    )
    log_group.add_argument(
        "-verbosity",
        type=int,
        default=1,
        help="How many epochs in between status updates",
    )
    log_group.add_argument(
        "-exp",
        type=str,
        default="test",
        help="Will group all run under this experiment name.",
    )
    log_group.add_argument(
        "-location",
        type=str,
        default="mlruns",
        help="Where to write mlflow experiment tracking stuff",
    )
    log_group.add_argument(
        "-run",
        type=str,
        default="neuromancer",
        help="Some name to tell what the experiment run was about.",
    )
    log_group.add_argument(
        "-logger",
        type=str,
        choices=["mlflow", "stdout"],
        default="stdout",
        help="Logging setup to use",
    )
    log_group.add_argument(
        "-train_visuals",
        action="store_true",
        help="Whether to create visuals, e.g. animations during training loop",
    )
    log_group.add_argument(
        "-trace_movie",
        action="store_true",
        help="Whether to plot an animation of the simulated and true dynamics",
    )

    return parser


def load_dataset(args, device, name):
    if systems[args.system] == "emulator":
        dataset = EmulatorDataset(
            system=args.system,
            nsim=args.nsim,
            norm=args.norm,
            nsteps=args.nsteps,
            device=device,
            savedir=args.savedir,
            seed=args.data_seed,
            name=name,
        )
        print(dataset.dims)
        # picking only ambient temp
        dataset.data['D'] = dataset.data['D'][:, [40]]
        dataset.shift_data['Dp'] = dataset.shift_data['Dp'][:, [40]]
        dataset.shift_data['Df'] = dataset.shift_data['Df'][:, [40]]
        dataset.dims['D'] = dataset.data['D'].shape
        dataset.dims['Dp'] = dataset.shift_data['Dp'].shape
        dataset.dims['Df'] = dataset.shift_data['Df'].shape
        dataset.min_max_norms['Dmin'] = dataset.min_max_norms['Dmin'][[40]]
        dataset.min_max_norms['Dmax'] = dataset.min_max_norms['Dmax'][[40]]

        dataset.nstep_data['Dp'] = dataset.nstep_data['Dp'][:, :, [40]]
        dataset.nstep_data['Df'] = dataset.nstep_data['Df'][:, :, [40]]
        dataset.train_data['Dp'] = dataset.train_data['Dp'][:, :, [40]]
        dataset.train_data['Df'] = dataset.train_data['Df'][:, :, [40]]
        dataset.dev_data['Dp'] = dataset.dev_data['Dp'][:, :, [40]]
        dataset.dev_data['Df'] = dataset.dev_data['Df'][:, :, [40]]
        dataset.test_data['Dp'] = dataset.test_data['Dp'][:, :, [40]]
        dataset.test_data['Df'] = dataset.test_data['Df'][:, :, [40]]

        dataset.train_loop['Dp'] = dataset.train_loop['Dp'][:, :, [40]]
        dataset.train_loop['Df'] = dataset.train_loop['Df'][:, :, [40]]
        dataset.dev_loop['Dp'] = dataset.dev_loop['Dp'][:, :, [40]]
        dataset.dev_loop['Df'] = dataset.dev_loop['Df'][:, :, [40]]
        dataset.test_loop['Dp'] = dataset.test_loop['Dp'][:, :, [40]]
        dataset.test_loop['Df'] = dataset.test_loop['Df'][:, :, [40]]

    else:
        dataset = FileDataset(
            system=args.system,
            nsim=args.nsim,
            norm=args.norm,
            nsteps=args.nsteps,
            device=device,
            savedir=args.savedir,
            name=name,
        )
    return dataset


def get_logger(args):
    if args.logger == "mlflow":
        logger = loggers.MLFlowLogger(
            args=args,
            savedir=args.savedir,
            verbosity=args.verbosity,
            stdout=(
                "nstep_dev_loss",
                "loop_dev_loss",
                "best_loop_dev_loss",
                "nstep_dev_ref_loss",
                "loop_dev_ref_loss",
            ),
        )

    else:
        logger = loggers.BasicLogger(
            args=args,
            savedir=args.savedir,
            verbosity=args.verbosity,
            stdout=(
                "nstep_dev_loss",
                "loop_dev_loss",
                "best_loop_dev_loss",
                "nstep_dev_ref_loss",
                "loop_dev_ref_loss",
            ),
        )
    return logger
