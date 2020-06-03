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

runs = mlflow.search_runs(experiment_ids=run_ids, filter_string="",
                          order_by=["metrics.bestopen ASC"])
runs = pd.DataFrame(runs)
runs.to_csv('nonlin_sysid_2020_5_31.csv')
runs.to_pickle('nonlin_sysid_2020_5_31.pkl')
runs = runs[['params.savedir', 'params.datafile', 'metrics.trainloss', 'metrics.train_reg',
             'metrics.devloss', 'metrics.dev_reg', 'metrics.open', 'metrics.bestopen',
             'metrics.nstep_train_loss', 'metrics.nstep_dev_loss', 'metrics.nstep_test_loss',
             'metrics.nstep_train_reg', 'metrics.nstep_dev_reg', 'metrics.nstep_test_reg',
             'metrics.open_train_loss', 'metrics.open_dev_loss', 'metrics.open_test_loss',
             'metrics.open_train_reg', 'metrics.open_dev_reg', 'metrics.open_test_reg']]
print(runs)


# # # # # #  OLD METRIC SYSTEM # # # # # # # # #
# runs = mlflow.search_runs(experiment_ids=run_ids, filter_string="",
#                           order_by=["metrics.open_dev_loss ASC"])
# runs = pd.DataFrame(runs)
# runs.to_csv('nonlin_sysid_2020_5_25.csv')
# runs.to_pickle('nonlin_sysid_2020_5_25.pkl')
# runs = runs[['params.savedir', 'params.datafile', 'metrics.open_dev_reg', 'metrics.open_test_loss',
#              'metrics.open_train_loss', 'metrics.nstep_dev_reg',
#              'metrics.nstep_test_loss', 'metrics.bestdev',
#              'metrics.open_test_reg', 'metrics.train_reg',
#              'metrics.nstep_train_loss', 'metrics.nstep_dev_loss', 'metrics.devloss',
#              'metrics.nstep_test_reg', 'metrics.open_train_reg', 'metrics.trainloss',
#              'metrics.nstep_train_reg', 'metrics.dev_reg', 'metrics.open_dev_loss']]
# print(runs)
