import mlflow
import os
import argparse
import pandas as pd


# Documentation for search API: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs
parser = argparse.ArgumentParser()
parser.add_argument('-expfolder', type=str, default='actuator_benchmark/mlruns')
args = parser.parse_args()
run_ids = [k for k in os.listdir(args.expfolder) if k.isdigit()]
runs_folder = 'mlruns'
mlflow.set_tracking_uri(args.expfolder)
runs = mlflow.search_runs(experiment_ids=run_ids, filter_string="",
                          order_by=["metrics.best_eval_mse"])
runs = pd.DataFrame(runs)
runs.to_csv('actuator_benchmark.csv')
runs.to_pickle('actuator_benchmark.pkl')
print(runs)