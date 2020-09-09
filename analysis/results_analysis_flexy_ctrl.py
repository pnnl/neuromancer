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



datapaths = ['./datasets/EED_building/EED_building.csv']
systems = ['building_EEd']
models = ['blocknlin', 'blackbox']
linear_map = ['linear', 'pf', 'softSVD']
nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
N = ['1','8', '16', '32', '64']
bias_onoff = ['-bias', '']
constrainted = ['unconstr', 'constr']


def process_res(res, key, system_metrics={}):
    # system_metrics[key] = {}
    if not res.empty:
        if res['metrics.best_loop_test_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_loop_test_loss'].idxmin()]
        else:
            best = None
        system_metrics[key]['best'] = best
        res = res.loc[res['metrics.best_loop_dev_loss'].notnull()]
        # extract metrics
        devnsteploss = res['metrics.best_nstep_dev_loss']
        devnsteploss = res['metrics.best_nstep_dev_loss']
        devopenloss = res['metrics.best_loop_dev_loss']
        testnsteploss = res['metrics.best_nstep_test_loss']
        testopenloss = res['metrics.best_loop_test_loss']
        trainnsteploss = res['metrics.best_nstep_train_loss']
        trainopenloss = res['metrics.best_loop_train_loss']
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

    res = pandas.read_pickle("../results_files/eed_sysid_2020_5_31.pkl")
    res.rename(columns={'params.ssm_type': 'models',
                        'params.linear_map': 'linear_map', 'params.nonlinear_map': 'nonlinear_map',
                        'params.nsteps': 'nsteps'}, inplace=True)
    # hierarchical index
    res.reset_index(inplace=True)
    res.set_index(['index', 'models', 'linear_map', 'nonlinear_map', 'nsteps'], drop=False, inplace=True)
    res.index

    system_metrics = {}
    if not res.empty:
        if res['metrics.best_loop_test_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_loop_test_loss'].idxmin()]
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
                for nonlinear in nonlinear_map:
                    system_metrics[nstep][type][linear][nonlinear] = {}
                    res_system_N_type_nonlin = res_system_N_type_lin.loc[res_system_N_type_lin.nonlinear_map == nonlinear]
                    system_metrics[nstep][type][linear] = process_res(res=res_system_N_type_nonlin, key=nonlinear,
                                                 system_metrics=system_metrics[nstep][type][linear])

    metrics_df = pandas.DataFrame.from_dict(system_metrics)

    # # # # # # # # # #
    # METRICS
    # # # # # # # # # #
    idx = []
    for model in models:
        for linear in linear_map:
            for nonlinear in nonlinear_map:
                idx.append(model+'_'+linear+'_'+nonlinear)
    stdopen, stdnstep, meanopen, meannstep, minopen, minnstep = \
        [pandas.DataFrame(index=idx,
          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            for linear in linear_map:
                for nonlinear in nonlinear_map:
                    # if not not metrics_df[i]:
                    #     if not not metrics_df[i][model][linear][nonlinear]:
                    stdopen.loc[model+'_'+linear+'_'+nonlinear,i] = \
                        metrics_df[i][model][linear][nonlinear]['std_test_openloss']
                    stdnstep.loc[model+'_'+linear+'_'+nonlinear,i] = \
                        metrics_df[i][model][linear][nonlinear]['std_test_nsteploss']
                    meanopen.loc[model+'_' + linear + '_' + nonlinear, i] = \
                        metrics_df[i][model][linear][nonlinear]['mean_test_openloss']
                    meannstep.loc[model+'_' + linear + '_' + nonlinear, i] = \
                        metrics_df[i][model][linear][nonlinear]['mean_test_nsteploss']
                    minopen.loc[model+'_' + linear + '_' + nonlinear, i] = \
                        metrics_df[i][model][linear][nonlinear]['min_test_openloss']
                    minnstep.loc[model+'_' + linear + '_' + nonlinear, i] = \
                        metrics_df[i][model][linear][nonlinear]['min_test_nsteploss']

    # # # # # # # # # #
    # PLOTS and Tables
    # # # # # # # # # #

    # Latex Table
    for k in [stdopen, stdnstep, meanopen, meannstep, minopen, minnstep]:
        print(k.to_latex(float_format=lambda x: '%.3f' % x))

    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.05
    ind = np.arange(len(N))
    for i, n in enumerate(idx):
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, minopen.loc[n], width, label=n, edgecolor='white')
    plt.xlabel('Training Prediction Horizon')
    plt.ylabel('Open loop MSE')
    plt.xticks(ind + width, N)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    plt.savefig('../figs/open_loop_'+model+'.eps')
    plt.savefig('../figs/open_loop_'+model+'.png')


    fig = plt.figure(figsize=(14, 10))
    ind = np.arange(len(N))
    for i, n in enumerate(idx):
        plt.bar(ind+i*width, minnstep.loc[n], width, label=n, edgecolor='white')
    plt.xlabel('Training Prediction Horizon')
    plt.ylabel('N-step MSE')
    plt.xticks(ind + 1.5*width,  N)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    plt.savefig('../figs/nstep_mse_'+model+'.eps')
    plt.savefig('../figs/nstep_mse_'+model+'.png')

    # copy artifacts
    if False:
        os.system('mkdir ../results_files/'+system_metrics['best']['params.savedir'])
        os.system('scp drgo694@ohmahgerd:'+system_metrics['best']['artifact_uri']+'/* '
                  +'../results_files/'+system_metrics['best']['params.savedir'])
