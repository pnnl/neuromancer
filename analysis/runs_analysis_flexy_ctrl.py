import mlflow
import os
import argparse
import pandas as pd

# Documentation for search API: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs
parser = argparse.ArgumentParser()
parser.add_argument('-expfolder', type=str, default='mlruns')

args = parser.parse_args()

run_ids = [k for k in os.listdir(args.expfolder) if k.isdigit()]
runs_folder = 'mlruns'
mlflow.set_tracking_uri(args.expfolder)

runs = mlflow.search_runs(experiment_ids=run_ids, filter_string="")
runs = pd.DataFrame(runs)
runs.to_csv('flexy_deepmpc_2020_9_8.csv')
runs.to_pickle('flexy_deepmpc_2020_9_8.pkl')
