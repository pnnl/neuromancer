"""
Results analysis
"""

import pandas
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os



datapaths = ['./nonlin_sysid_2021_2_7.csv']
models = ['hammerstein', 'hw']
linear_map = ['linear', 'pf', 'softSVD']
# nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
N = ['8',  '32']
bias_onoff = ['-bias', '']
hidden_states = [5, 10]
constrainted = ['unconstr', 'constr']
weights = ['params.Q_con_x', 'params.Q_dx', 'params.Q_con_fdu', 'params.Q_sub', 'params.Q_y']

def process_res(res, key, system_metrics={}):
    # system_metrics[key] = {}
    if not res.empty:
        if res['metrics.best_loop_test_ref_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_loop_test_ref_loss'].idxmin()]
        else:
            best = None
        system_metrics[key]['best'] = best
        res = res.loc[res['metrics.best_loop_dev_ref_loss'].notnull()]
        # extract metrics
        devnsteploss = res['metrics.best_nstep_dev_ref_loss']
        devopenloss = res['metrics.best_loop_dev_ref_loss']
        testnsteploss = res['metrics.best_nstep_test_ref_loss']
        testopenloss = res['metrics.best_loop_test_ref_loss']
        trainnsteploss = res['metrics.best_nstep_train_ref_loss']
        trainopenloss = res['metrics.best_loop_train_ref_loss']
        # log metrics
        system_metrics[key]['mean_dev_openloss'] = devopenloss.mean()
        system_metrics[key]['mean_dev_nsteploss'] = devnsteploss.mean()
        system_metrics[key]['mean_test_openloss'] = testopenloss.mean()
        system_metrics[key]['mean_test_nsteploss'] = testnsteploss.mean()
        system_metrics[key]['std_dev_openloss'] = devopenloss.std()
        system_metrics[key]['std_dev_nsteploss'] = devnsteploss.std()
        system_metrics[key]['std_test_openloss'] = testopenloss.std()
        system_metrics[key]['std_test_nsteploss'] = testnsteploss.std()
        system_metrics[key]['min_dev_openloss'] = devopenloss.min()
        system_metrics[key]['min_dev_nsteploss'] = devnsteploss.min()
        system_metrics[key]['min_test_openloss'] = testopenloss.min()
        system_metrics[key]['min_test_nsteploss'] = testnsteploss.min()
        return system_metrics

if __name__ == '__main__':

    res = pandas.read_pickle("./nonlin_sysid_2021_2_7.pkl")
    res.rename(columns={'params.Q_con_x': 'Q_con_x',
                        'params.linear_map': 'linear_map',
                        'params.ssm_type': 'models',
                        'params.nsteps': 'nsteps'}, inplace=True)
    # hierarchical index
    res.reset_index(inplace=True)
    res.set_index(['index', 'linear_map', 'models', 'nsteps', 'Q_con_x'], drop=False, inplace=True)
    res.index

    system_metrics = {}
    if not res.empty:
        if res['metrics.best_loop_dev_ref_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_loop_dev_ref_loss'].idxmin()]
        else:
            best = None
        system_metrics['best'] = best
    for nstep in N:
        system_metrics[nstep] = {}
        res_system_N = res.loc[res.nsteps == nstep]
        system_metrics = process_res(res=res_system_N, key=nstep, system_metrics=system_metrics)
        for type in models:
            system_metrics[nstep][type] = {}
            res_system_N_type = res_system_N.loc[res_system_N.models == type]
            system_metrics[nstep] = process_res(res=res_system_N_type, key=type, system_metrics=system_metrics[nstep])
            for linear in linear_map:
                system_metrics[nstep][type][linear] = {}
                res_system_N_type_lin = res_system_N_type.loc[res_system_N_type.linear_map == linear]
                system_metrics[nstep][type] = process_res(res=res_system_N_type_lin, key=linear, system_metrics=system_metrics[nstep][type])

    metrics_df = pandas.DataFrame.from_dict(system_metrics)

