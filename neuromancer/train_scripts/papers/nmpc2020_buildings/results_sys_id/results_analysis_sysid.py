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

    # best black box model
    res_system_N = res.loc[res.nsteps == '16']
    res_black_box = res_system_N.loc[res_system_N.models == 'blackbox']
    best_res_black_box = res_black_box.loc[res_black_box['metrics.best_loop_test_loss'].idxmin()]
    best_res_black_box['artifact_uri']

    # # # # # # # # # #
    # METRICS
    # # # # # # # # # #
    idx = []
    for model in models:
        for linear in linear_map:
            idx.append(model+'_'+linear)
    stdopen, stdnstep, meanopen, meannstep, minopen, minnstep = \
        [pandas.DataFrame(index=idx,
          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            for linear in linear_map:
                stdopen.loc[model+'_'+linear,i] = \
                    metrics_df[i][model][linear]['std_test_openloss']
                stdnstep.loc[model+'_'+linear,i] = \
                    metrics_df[i][model][linear]['std_test_nsteploss']
                meanopen.loc[model+'_' + linear, i] = \
                    metrics_df[i][model][linear]['mean_test_openloss']
                meannstep.loc[model+'_' + linear, i] = \
                    metrics_df[i][model][linear]['mean_test_nsteploss']
                minopen.loc[model+'_' + linear, i] = \
                    metrics_df[i][model][linear]['min_test_openloss']
                minnstep.loc[model+'_' + linear, i] = \
                    metrics_df[i][model][linear]['min_test_nsteploss']

    idx_lin = []
    for model in models:
        for linear in linear_map:
            idx_lin.append(model+'_'+linear)
    stdopen_lin, stdnstep_lin, meanopen_lin, \
    meannstep_lin, minopen_lin, minnstep_lin = \
        [pandas.DataFrame(index=idx_lin,
          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            for linear in linear_map:
                temp = [minopen.loc[model + '_' + linear , i] ]
                minopen_lin.loc[model+'_'+linear,i] = min(temp)
                temp = [minnstep.loc[model + '_' + linear, i]]
                minnstep_lin.loc[model+'_'+linear,i] = min(temp)

    for k in [minopen_lin, minnstep_lin]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    idx_nlin = []
    for model in models:
        idx_nlin.append(model)
    stdopen_nlin, stdnstep_nlin, meanopen_nlin, \
    meannstep_nlin, minopen_nlin, minnstep_nlin = \
        [pandas.DataFrame(index=idx_nlin,
                          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            temp = [minopen.loc[model + '_' + linear , i] for linear in linear_map]
            minopen_nlin.loc[model , i] = min(temp)
            temp = [minnstep.loc[model + '_' + linear, i] for linear in linear_map]
            minnstep_nlin.loc[model , i] = min(temp)

    for k in [minopen_nlin, minnstep_nlin]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    idx_reduce = []
    for model in models:
        idx_reduce.append(model)
    stdopen_reduce, stdnstep_reduce, meanopen_reduce, \
    meannstep_reduce, minopen_reduce, minnstep_reduce = \
        [pandas.DataFrame(index=idx_reduce,
                          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            temp = [minopen.loc[model + '_' + linear, i] for linear in linear_map]
            minopen_reduce.loc[model, i] = min(temp)
            temp = [minnstep.loc[model + '_' + linear , i] for linear in linear_map]
            minnstep_reduce.loc[model, i] = min(temp)

    for k in [minopen_reduce, minnstep_reduce]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.05
    ind = np.arange(len(N))
    for i, n in enumerate(idx):
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



    idx_nlin_labels = ['struct. mlp', 'struct. resnet', 'struct. rnn',
                      'unstruct. mlp', 'unstruct. resnet', 'unstruct. rnn']
    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.15
    ind = np.arange(len(N))
    for i, n in enumerate(idx_nlin):
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, minopen_nlin.loc[n], width, label=idx_nlin_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('open-loop MSE', fontsize=24)
    plt.xticks(ind + width, N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.savefig('../figs/open_loop_'+model+'_nlin.eps')
    plt.savefig('../figs/open_loop_'+model+'_nlin.png')

    fig = plt.figure(figsize=(14, 10))
    ind = np.arange(len(N))
    for i, n in enumerate(idx_nlin):
        plt.bar(ind+i*width, minnstep_nlin.loc[n], width, label=idx_nlin_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('N-step MSE', fontsize=24)
    plt.xticks(ind + 1.5*width,  N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.savefig('../figs/nstep_mse_'+model+'_nlin.eps')
    plt.savefig('../figs/nstep_mse_'+model+'_nlin.png')


