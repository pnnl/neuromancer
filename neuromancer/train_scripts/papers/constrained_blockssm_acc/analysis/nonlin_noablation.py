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
    default="no_ablate.sh",
)
parser.add_argument(
    "-nsamples",
    type=int,
    help="Number of samples for each experimental configuration",
    default=5,
)
parser.add_argument("-params_file", default="/qfs/projects/deepmpc/best_systems.csv")

args = parser.parse_args()

params = pd.read_csv(args.params_file)
stub = (
    "python system_id.py -gpu 0 "
    + f"-epochs 15000 -location {args.results} -logger mlflow "
    + "-warmup 100 -patience 100 -exp noablate "
)

with open(args.exp_script, "w") as f:
    for j in params.index:
        for i in range(args.nsamples):
            exp = params["system"][j]
            savedir = "_".join([params["system"][j], str(i)])
            argstring = " ".join(
                ["-" + p + " " + str(params[p][j]) for p in params.columns]
                + [f"-savedir {savedir}"]
            )
            cmd = stub + argstring
            f.write(f"{cmd}\n")