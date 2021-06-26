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
# linear_map = ['linear', 'pf', 'softSVD']
linear_map = ['linear', 'pf']
nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
# nonlinear_map = ['mlp', 'rnn']
# N = ['1','8', '16', '32', '64']
N = ['8', '16', '32', '64']
bias_onoff = ['-bias', '']
constrainted = ['unconstr', 'constr']
weights = ['params.Q_con_x', 'params.Q_dx', 'params.Q_con_fdu', 'params.Q_sub', 'params.Q_y']

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
    res.rename(columns={'params.ssm_type': 'models', 'params.Q_con_x': 'Q_con_x',
                        'params.linear_map': 'linear_map', 'params.nonlinear_map': 'nonlinear_map',
                        'params.nsteps': 'nsteps'}, inplace=True)
    # hierarchical index
    res.reset_index(inplace=True)
    res.set_index(['index', 'models', 'linear_map', 'nonlinear_map', 'nsteps', 'Q_con_x'], drop=False, inplace=True)
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
                    for constr in constrainted:
                        system_metrics[nstep][type][linear][nonlinear][constr] = {}
                        if constr == 'constr':
                            res_system_N_type_nonlin_con = res_system_N_type_nonlin[
                                res_system_N_type_nonlin['Q_con_x'] == '1.0']
                        else:
                            res_system_N_type_nonlin_con = res_system_N_type_nonlin[
                                res_system_N_type_nonlin['Q_con_x'] != '1.0']
                        system_metrics[nstep][type][linear][nonlinear] = process_res(res=res_system_N_type_nonlin_con, key=constr,
                                                                          system_metrics=system_metrics[nstep][type][linear][nonlinear])

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
            for nonlinear in nonlinear_map:
                # idx.append(model + '_' + linear + '_' + nonlinear)
                for constr in constrainted:
                    idx.append(model+'_'+linear+'_'+nonlinear+'_'+constr)
    stdopen, stdnstep, meanopen, meannstep, minopen, minnstep, constr_weight = \
        [pandas.DataFrame(index=idx,
          columns=N) for i in range(7)]

    for i in N:
        for model in models:
            for constr in constrainted:
                for linear in linear_map:
                    for nonlinear in nonlinear_map:
                        stdopen.loc[model+'_'+linear+'_'+nonlinear+'_'+constr,i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['std_test_openloss']
                        stdnstep.loc[model+'_'+linear+'_'+nonlinear+'_'+constr,i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['std_test_nsteploss']
                        meanopen.loc[model+'_' + linear + '_' + nonlinear+'_'+constr, i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['mean_test_openloss']
                        meannstep.loc[model+'_' + linear + '_' + nonlinear+'_'+constr, i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['mean_test_nsteploss']
                        minopen.loc[model+'_' + linear + '_' + nonlinear+'_'+constr, i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['min_test_openloss']
                        minnstep.loc[model+'_' + linear + '_' + nonlinear+'_'+constr, i] = \
                            metrics_df[i][model][linear][nonlinear][constr]['min_test_nsteploss']
                        # constr_weight.loc[model+'_' + linear + '_' + nonlinear+'_'+constr, i] = \
                        #     0 if metrics_df[i][model][linear][nonlinear][constr]['best']['Q_con_x'] is None else 1

    idx_lin = []
    for model in models:
        for linear in linear_map:
            idx_lin.append(model+'_'+linear)
    stdopen_lin, stdnstep_lin, meanopen_lin, \
    meannstep_lin, minopen_lin, minnstep_lin, constr_weight_lin = \
        [pandas.DataFrame(index=idx_lin,
          columns=N) for i in range(7)]

    for i in N:
        for model in models:
            for linear in linear_map:
                temp = [minopen.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for nonlinear in nonlinear_map for constr in constrainted]
                minopen_lin.loc[model+'_'+linear,i] = min(temp)
                temp = [minnstep.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for nonlinear in nonlinear_map for constr in constrainted]
                minnstep_lin.loc[model+'_'+linear,i] = min(temp)

    for k in [minopen_lin, minnstep_lin]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    idx_nlin = []
    for model in models:
        for nonlinear in nonlinear_map:
            if not (nonlinear == "residual_mlp" and model == "blocknlin"):
                idx_nlin.append(model + '_' + nonlinear)
    stdopen_nlin, stdnstep_nlin, meanopen_nlin, \
    meannstep_nlin, minopen_nlin, minnstep_nlin, constr_weight_nlin = \
        [pandas.DataFrame(index=idx_nlin,
                          columns=N) for i in range(7)]

    for i in N:
        for model in models:
            for nonlinear in nonlinear_map:
                if not (nonlinear == "residual_mlp" and model == "blocknlin"):
                    temp = [minopen.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for constr in
                            constrainted for linear in linear_map]
                    minopen_nlin.loc[model + '_' + nonlinear, i] = min(temp)
                    temp = [minnstep.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for constr in
                            constrainted for linear in linear_map]
                    minnstep_nlin.loc[model + '_' + nonlinear, i] = min(temp)

    for k in [minopen_nlin, minnstep_nlin]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    idx_reduce = []
    for model in models:
        for constr in constrainted:
            idx_reduce.append(model + '_' + constr)
    stdopen_reduce, stdnstep_reduce, meanopen_reduce, \
    meannstep_reduce, minopen_reduce, minnstep_reduce, constr_weight_reduce = \
        [pandas.DataFrame(index=idx_reduce,
                          columns=N) for i in range(7)]

    for i in N:
        for model in models:
            for constr in constrainted:
                temp = [minopen.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for nonlinear in
                        nonlinear_map for linear in linear_map]
                minopen_reduce.loc[model + '_' + constr, i] = min(temp)
                temp = [minnstep.loc[model + '_' + linear + '_' + nonlinear + '_' + constr, i] for nonlinear in
                        nonlinear_map for linear in linear_map]
                minnstep_reduce.loc[model + '_' + constr, i] = min(temp)

    for k in [minopen_reduce, minnstep_reduce]:
        print(k.to_latex(float_format=lambda x: '%.5f' % x))


    # for i in N:
    #     for model in models:
    #         for linear in linear_map:
    #             for nonlinear in nonlinear_map:
    #                 # if not not metrics_df[i]:
    #                 #     if not not metrics_df[i][model][linear][nonlinear]:
    #                 stdopen.loc[model+'_'+linear+'_'+nonlinear,i] = \
    #                     metrics_df[i][model][linear][nonlinear]['std_test_openloss']
    #                 stdnstep.loc[model+'_'+linear+'_'+nonlinear,i] = \
    #                     metrics_df[i][model][linear][nonlinear]['std_test_nsteploss']
    #                 meanopen.loc[model+'_' + linear + '_' + nonlinear, i] = \
    #                     metrics_df[i][model][linear][nonlinear]['mean_test_openloss']
    #                 meannstep.loc[model+'_' + linear + '_' + nonlinear, i] = \
    #                     metrics_df[i][model][linear][nonlinear]['mean_test_nsteploss']
    #                 minopen.loc[model+'_' + linear + '_' + nonlinear, i] = \
    #                     metrics_df[i][model][linear][nonlinear]['min_test_openloss']
    #                 minnstep.loc[model+'_' + linear + '_' + nonlinear, i] = \
    #                     metrics_df[i][model][linear][nonlinear]['min_test_nsteploss']
    #                 constr_weight.loc[model+'_' + linear + '_' + nonlinear, i] = \
    #                     0 if metrics_df[i][model][linear][nonlinear]['best']['Q_con_x'] is None else 1

    # # # # # # # # # #
    # PLOTS and Tables
    # # # # # # # # # #

    # # Latex Table
    # for k in [stdopen, stdnstep, meanopen, meannstep, minopen, minnstep]:
    #     print(k.to_latex(float_format=lambda x: '%.3f' % x))



    # # constrtaints
    # fig = plt.figure(figsize=(14, 10))
    # width = 0.05
    # ind = np.arange(len(N))
    # for i, n in enumerate(idx):
    #     # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
    #     plt.bar(ind+i*width, constr_weight.loc[n], width, label=n, edgecolor='white')
    # plt.xlabel('Training Prediction Horizon')
    # plt.ylabel('constraints on-off')
    # plt.xticks(ind + width, N)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.grid()
    # plt.yscale("log")
    # plt.savefig('../figs/constr_'+model+'.eps')
    # plt.savefig('../figs/constr_'+model+'.png')


    # # Bar plot
    # fig = plt.figure(figsize=(14, 10))
    # width = 0.05
    # ind = np.arange(len(N))
    # for i, n in enumerate(idx):
    #     # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
    #     plt.bar(ind+i*width, minopen.loc[n], width, label=n, edgecolor='white')
    # plt.xlabel('Training Prediction Horizon')
    # plt.ylabel('Open loop MSE')
    # plt.xticks(ind + width, N)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.grid()
    # plt.yscale("log")
    # plt.savefig('../figs/open_loop_'+model+'.eps')
    # plt.savefig('../figs/open_loop_'+model+'.png')
    #
    # fig = plt.figure(figsize=(14, 10))
    # ind = np.arange(len(N))
    # for i, n in enumerate(idx):
    #     plt.bar(ind+i*width, minnstep.loc[n], width, label=n, edgecolor='white')
    # plt.xlabel('Training Prediction Horizon')
    # plt.ylabel('N-step MSE')
    # plt.xticks(ind + 1.5*width,  N)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.grid()
    # plt.yscale("log")
    # plt.savefig('../figs/nstep_mse_'+model+'.eps')
    # plt.savefig('../figs/nstep_mse_'+model+'.png')

    model = 'blocknlin'

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Effect of neural blocks architecture on open-loop andN-step ahead MSE
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # idx_nlin_labels = ['struct. mlp', 'struct. rnn',
    #                   'unstruct. mlp', 'unstruct. rnn']
    idx_nlin_labels = ['struct. mlp',  'struct. rnn',
                      'unstruct. mlp', 'unstruct. resnet', 'unstruct. rnn']
    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.15
    ind = np.arange(len(N))
    ymax = 0
    for i, n in enumerate(idx_nlin):
        if ymax < minopen_nlin.loc[n].max():
            ymax = minopen_nlin.loc[n].max()
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, minopen_nlin.loc[n], width, label=idx_nlin_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('open-loop MSE', fontsize=24)
    plt.xticks(ind + 1.5*width, N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    # plt.yscale("log")
    plt.ylim([0.008, 1.15 * ymax])
    plt.rc('ytick', labelsize=22)
    plt.tight_layout()
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
    # plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.ylim([0.008, 1.15 * ymax])
    plt.tight_layout()
    plt.savefig('../figs/nstep_mse_'+model+'_nlin.eps')
    plt.savefig('../figs/nstep_mse_'+model+'_nlin.png')


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Effect of eigenvalue constraints viapffactorization on open-loop andN-step ahead MSE
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    idx_lin_labels = ['struct. linear weight', 'struct. pf weight',
                      'unstruct. linear weight', 'unstruct. pf weight']
    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.15
    ind = np.arange(len(N))
    ymax = 0
    for i, n in enumerate(idx_lin):
        if ymax < minopen_lin.loc[n].max():
            ymax = minopen_lin.loc[n].max()
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, minopen_lin.loc[n], width, label=idx_lin_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('open-loop MSE', fontsize=24)
    plt.xticks(ind + 1.5*width, N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    # plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.ylim([0.008, 1.15 * ymax])
    plt.tight_layout()
    plt.savefig('../figs/open_loop_'+model+'_lin.eps')
    plt.savefig('../figs/open_loop_'+model+'_lin.png')

    fig = plt.figure(figsize=(14, 10))
    ind = np.arange(len(N))
    for i, n in enumerate(idx_lin):
        plt.bar(ind+i*width, minnstep_lin.loc[n], width, label=idx_lin_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('N-step MSE', fontsize=24)
    plt.xticks(ind + 1.5*width,  N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    # plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.ylim([0.008, 1.15 * ymax])
    plt.tight_layout()
    plt.savefig('../figs/nstep_mse_'+model+'_lin.eps')
    plt.savefig('../figs/nstep_mse_'+model+'_lin.png')


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Effect of penalty constraints on open-loop andN-step ahead MSE
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    idx_reduce_labels = ['struct. unconstr.', 'struct. constr.', 'unstruct. unconstr.', 'unstruct. constr.']

    # Bar plot
    fig = plt.figure(figsize=(14, 10))
    width = 0.15
    ind = np.arange(len(N))
    ymax = 0
    for i, n in enumerate(idx_reduce):
        if ymax < minopen_reduce.loc[n].max():
            ymax = minopen_reduce.loc[n].max()
        # plt.bar(ind + i * width, minopen.iloc[i], width, label=n, edgecolor='white')
        plt.bar(ind+i*width, minopen_reduce.loc[n], width, label=idx_reduce_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('open-loop MSE', fontsize=24)
    plt.xticks(ind + 1.5*width, N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    # plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.ylim([0.008, 1.15 * ymax])
    plt.tight_layout()
    plt.savefig('../figs/open_loop_'+model+'_reduce.eps')
    plt.savefig('../figs/open_loop_'+model+'_reduce.png')

    fig = plt.figure(figsize=(14, 10))
    ind = np.arange(len(N))
    for i, n in enumerate(idx_reduce):
        plt.bar(ind+i*width, minnstep_reduce.loc[n], width, label=idx_reduce_labels[i], edgecolor='white', alpha=0.9)
    plt.xlabel('prediction horizon', fontsize=24)
    plt.ylabel('N-step MSE', fontsize=24)
    plt.xticks(ind + 1.5*width,  N, fontsize=22)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    plt.grid()
    # plt.yscale("log")
    plt.rc('ytick', labelsize=22)
    plt.ylim([0.008, 1.15 * ymax])
    plt.tight_layout()
    plt.savefig('../figs/nstep_mse_'+model+'_reduce.eps')
    plt.savefig('../figs/nstep_mse_'+model+'_reduce.png')


    normalizations = {'Ymin': -1.3061713953490333, 'Ymax': 32.77003662201578,
                      'Umin': -2.1711117, 'Umax': 33.45899931,
                      'Dmin': 29.46308055, 'Dmax': 48.97325791}

    scaling = normalizations['Ymax']-normalizations['Ymin']
    minopen_reduce*scaling
    minnstep_reduce*scaling


    # copy artifacts
    if False:
        os.system('mkdir ../results_files/'+system_metrics['best']['params.savedir'])
        os.system('scp drgo694@ohmahgerd:'+system_metrics['best']['artifact_uri']+'/* '
                  +'../results_files/'+system_metrics['best']['params.savedir'])
