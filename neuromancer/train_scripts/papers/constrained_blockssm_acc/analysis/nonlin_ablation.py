import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-results",
    type=str,
    help="Where to log mlflow results",
    default="mlruns",
)
parser.add_argument(
    "-exp_script",
    type=str,
    help="Where to save sbatch scripts and log files",
    default="individual_ablate.sh",
)
parser.add_argument(
    "-nsamples",
    type=int,
    help="Number of samples for each experimental configuration",
    default=5,
)
parser.add_argument("-params_file", default="/qfs/projects/deepmpc/best_systems.csv")

args = parser.parse_args()

keys = [
    "linear_map",
    "estimator_input_window",
    "sigma_max",
    "n_layers",
    "state_estimator",
    "ssm_type",
    "nonlinear_map",
    "system",
    "activation",
    "nx_hidden",
]

ablations = [
    ("Q_dx", 0.0),
    ("Q_con_x", 0.0),
    ("Q_con_fdu", 0.0),
    ("linear_map", "linear"),
]

params = pd.read_csv(args.params_file)
stub = (
    "python system_id.py -gpu 0 -epochs 15000 "
    + f"-location {args.results} -logger mlflow "
    + "-warmup 100 -patience 100 "
)

with open(args.exp_script, "w") as f:
    for j in params.index:
        for i in range(args.nsamples):
            for ab in ablations:
                params_copy = pd.read_csv(args.params_file)
                params_copy.loc[j, [ab[0]]] = ab[1]
                exp = params_copy["system"][j]
                savedir = "_".join([params_copy["system"][j], str(i), ab[0]])
                argstring = " ".join(
                    ["-" + p + " " + str(params_copy[p][j]) for p in params_copy.columns]
                    + [f"-savedir {savedir}"]
                )
                cmd = stub + argstring
                f.write(f"{cmd}\n")