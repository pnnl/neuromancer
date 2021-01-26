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
    default="all_ablate.sh",
)
parser.add_argument(
    "-nsamples",
    type=int,
    help="Number of samples for each experimental configuration",
    default=5,
)
parser.add_argument("-params_file", default="best_systems.csv")

args = parser.parse_args()

params = pd.read_csv(args.params_file)

ablations = [
    ("Q_dx", 0.0),
    ("Q_con_x", 0.0),
    ("Q_con_fdu", 0.0),
    ("linear_map", "linear"),
]

stub = (
    "python system_id.py -gpu 0 "
    + f"-epochs 15000 -location {args.results} -logger mlflow "
    + "-exp allablate -warmup 100 -patience 100 "
)

for j in params.index:
    for ab in ablations:
        params.loc[j, [ab[0]]] = ab[1]

with open(args.exp_script, "w") as f:
    for j in params.index:
        for i in range(args.nsamples):
            exp = params["system"][j]
            savedir = "_".join(["all_ablate", params["system"][j], str(i)])
            argstring = " ".join(
                ["-" + p + " " + str(params[p][j]) for p in params.columns]
                + [f"-savedir {savedir}"]
            )
            cmd = stub + argstring
            f.write(f"{cmd}\n")