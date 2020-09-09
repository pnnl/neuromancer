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



systems = ['flexy air ctrl']


linear_map = ['linear', 'pf', 'softSVD']
N = [4, 8, 16]
policy = ['mlp', 'rnn']
activations = ['gelu', 'softexp']
models = ['/people/drgo694/neuromancer/neuromancer/neuromancer/datasets/Flexy_air/best_model_flexy1.pth',
          '/qfs/projects/deepmpc/best_flexy_models/best_blocknlin_nlinsearch/best_model.pth',
          '/people/drgo694/neuromancer/neuromancer/neuromancer/datasets/Flexy_air/best_model_flexy2.pth']


def process_res(res, key, system_metrics={}):
    # system_metrics[key] = {}
    if not res.empty:
        if res['metrics.best_nstep_test_ref_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_nstep_test_ref_loss'].idxmin()]
        else:
            best = None
        system_metrics[key]['best'] = best
        res = res.loc[res['metrics.best_nstep_dev_ref_loss'].notnull()]
        # extract metrics
        nstep_dev = res['metrics.best_nstep_dev_loss']
        nstep_test = res['metrics.best_nstep_test_loss']
        nstep_test_ref = res['metrics.best_nstep_test_ref_loss']
        nstep_dev_ref = res['metrics.best_nstep_dev_ref_loss']

        # log metrics
        system_metrics[key]['mean_nstep_dev'] = nstep_dev.mean()
        system_metrics[key]['mean_nstep_test'] = nstep_test.mean()
        system_metrics[key]['mean_nstep_test_ref'] = nstep_test_ref.mean()
        system_metrics[key]['mean_nstep_dev_ref'] = nstep_dev_ref.mean()

        system_metrics[key]['std_nstep_dev'] = nstep_dev.std()
        system_metrics[key]['std_nstep_test'] = nstep_test.std()
        system_metrics[key]['std_nstep_test_ref'] = nstep_test_ref.std()
        system_metrics[key]['std_nstep_dev_ref'] = nstep_dev_ref.std()

        system_metrics[key]['min_nstep_dev'] = nstep_dev.min()
        system_metrics[key]['min_nstep_test'] = nstep_test.min()
        system_metrics[key]['min_nstep_test_ref'] = nstep_test_ref.min()
        system_metrics[key]['min_nstep_dev_ref'] = nstep_dev_ref.min()
        return system_metrics

if __name__ == '__main__':

    # res = pandas.read_pickle("../results/flexy_deepmpc_2020_9_8.pkl")
    res = pandas.read_csv("../results/flexy_deepmpc_2020_9_8.csv")

    res.rename(columns={'params.model_file': 'models',  'params.linear_map': 'linear_map',
                        'params.activation': 'activation', 'params.policy': 'policy',
                        'params.nsteps': 'nsteps'}, inplace=True)
    # hierarchical index
    res.reset_index(inplace=True)
    res.set_index(['index', 'models', 'linear_map', 'activation', 'policy', 'nsteps'], drop=False, inplace=True)
    res.index

    system_metrics = {}
    if not res.empty:
        if res['metrics.best_nstep_dev_ref_loss'].idxmin() is not np.nan:
            best = res.loc[res['metrics.best_nstep_dev_ref_loss'].idxmin()]
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
                for nonlinear in policy:
                    system_metrics[nstep][type][linear][nonlinear] = {}
                    res_system_N_type_nonlin = res_system_N_type_lin.loc[res_system_N_type_lin.policy == nonlinear]
                    system_metrics[nstep][type][linear] = process_res(res=res_system_N_type_nonlin, key=nonlinear,
                                                 system_metrics=system_metrics[nstep][type][linear])

    metrics_df = pandas.DataFrame.from_dict(system_metrics)

    # # # # # # # # # #
    # METRICS
    # # # # # # # # # #
    idx = []
    for model in models:
        idx.append(model)
    stdopen, stdnstep, meanopen, meannstep, min_nstep_dev, min_nstep_test = \
        [pandas.DataFrame(index=idx,
          columns=N) for i in range(6)]

    for i in N:
        for model in models:
            # if not not metrics_df[i]:
            #     if not not metrics_df[i][model][linear][nonlinear]:
            stdopen.loc[model,i] = \
                metrics_df[i][model]['std_nstep_dev_ref']
            stdnstep.loc[model,i] = \
                metrics_df[i][model]['std_nstep_test_ref']
            meanopen.loc[model, i] = \
                metrics_df[i][model]['mean_nstep_dev_ref']
            meannstep.loc[model, i] = \
                metrics_df[i][model]['mean_nstep_test_ref']
            min_nstep_dev.loc[model, i] = \
                metrics_df[i][model]['min_nstep_dev_ref']
            min_nstep_test.loc[model, i] = \
                metrics_df[i][model]['min_nstep_test_ref']

    # # # # # # # # # #
    # PLOTS and Tables
    # # # # # # # # # #

    # Latex Table
    for k in [stdopen, stdnstep, meanopen, meannstep, min_nstep_dev, min_nstep_test]:
        print(k.to_latex(float_format=lambda x: '%.3f' % x))

    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.05
    ind = np.arange(len(N))
    for i, n in enumerate(idx):
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, min_nstep_dev.loc[n], width, label=n, edgecolor='white')
    plt.xlabel('Training Prediction Horizon')
    plt.ylabel('min nstep dev MSE')
    plt.xticks(ind + width, N)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    # plt.savefig('../figs/open_loop.eps')
    # plt.savefig('../figs/open_loop_.png')


    fig = plt.figure(figsize=(14, 10))
    ind = np.arange(len(N))
    for i, n in enumerate(idx):
        plt.bar(ind+i*width, min_nstep_test.loc[n], width, label=n, edgecolor='white')
    plt.xlabel('Training Prediction Horizon')
    plt.ylabel('min nstep test MSE')
    plt.xticks(ind + 1.5*width,  N)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.yscale("log")
    # plt.savefig('../figs/nstep_mse.eps')
    # plt.savefig('../figs/nstep_mse.png')

    # copy artifacts
    if False:
        os.system('mkdir ..\\results\\'+system_metrics['best']['params.savedir'].split('.')[0].split('/')[1])
        os.system('scp drgo694@ohmahgerd:'+system_metrics['best']['artifact_uri']+'/* '+'..\\results\\'+system_metrics['best']['params.savedir'].split('.')[0].split('/')[1])
