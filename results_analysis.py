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

datapaths = ['./datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat',
                 './datasets/NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat',
                 './datasets/NLIN_MIMO_CSTR/NLIN_MIMO_CSTR2.mat',
                 './datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat']
systems = ['tank','vehicle3','reactor','aero']
ssm_type=['BlockSSM', 'BlackSSM']
linear_map=['pf', 'linear']
nonlinear_map= ['linear', 'sparse_residual_mlp', 'mlp']
N = ['2', '4', '8', '16', '32']

res = pandas.read_pickle("./results_files/nonlin_sysid_2020_5_14.pkl")
res.rename(columns={'params.datafile':'datafile','params.ssm_type':'ssm_type',
                    'params.linear_map':'linear_map','params.nonlinear_map':'nonlinear_map',
                    'params.nsteps':'nsteps'}, inplace=True)

for dpath, system in zip(datapaths, systems):
    res.loc[:,'datafile'].replace(dpath, system, inplace=True)

# hierarchical index
res.reset_index(inplace=True)
res.set_index(['index','datafile','ssm_type','linear_map',
               'nonlinear_map','nsteps'], drop=False, inplace=True)
res.index
res.head()

# select best models
system_metrics = {}
for system in systems:
    system_metrics[system] = {}
    res_system = res.loc[res.datafile==system,:]
    if res_system['metrics.open_test_loss'].idxmin() is not np.nan:
        best = res_system.loc[res_system['metrics.open_test_loss'].idxmin()]
    else:
        best = None
    res_system = res_system.loc[res_system['metrics.open_dev_loss'].notnull()]
    devopenloss = res_system['metrics.open_dev_loss']
    testnsteploss = res_system['metrics.nstep_test_loss']
    testopenloss = res_system['metrics.open_test_loss']
    system_metrics[system]['best'] = best
    system_metrics[system]['mean_dev_openloss'] = devopenloss.mean()
    system_metrics[system]['mean_test_openloss'] = testopenloss.mean()
    system_metrics[system]['mean_test_nsteploss'] = testnsteploss.mean()
    system_metrics[system]['std_dev_openloss'] = devopenloss.std()
    system_metrics[system]['std_test_openloss'] = testopenloss.std()
    system_metrics[system]['std_test_nsteploss'] = testnsteploss.std()
    system_metrics[system]['min_dev_openloss'] = devopenloss.min()
    system_metrics[system]['min_test_openloss'] = testopenloss.min()
    system_metrics[system]['min_test_nsteploss'] = testnsteploss.min()

    for nstep in N:
        system_metrics[system][nstep] = {}
        res_system_N = res_system.loc[res_system.nsteps == nstep]
        if not res_system_N.empty:
            if res_system_N['metrics.open_test_loss'].idxmin() is not np.nan:
                best = res_system_N.loc[res_system_N['metrics.open_test_loss'].idxmin()]
            else:
                best = None
            res_system_N = res_system_N.loc[res_system_N['metrics.open_dev_loss'].notnull()]
            devopenloss = res_system_N['metrics.open_dev_loss']
            testnsteploss = res_system_N['metrics.nstep_test_loss']
            testopenloss = res_system_N['metrics.open_test_loss']
            system_metrics[system][nstep]['best'] = best
            system_metrics[system][nstep]['mean_dev_openloss'] = devopenloss.mean()
            system_metrics[system][nstep]['mean_test_openloss'] = testopenloss.mean()
            system_metrics[system][nstep]['mean_test_nsteploss'] = testnsteploss.mean()
            system_metrics[system][nstep]['std_dev_openloss'] = devopenloss.std()
            system_metrics[system][nstep]['std_test_openloss'] = testopenloss.std()
            system_metrics[system][nstep]['std_test_nsteploss'] = testnsteploss.std()
            system_metrics[system][nstep]['min_dev_openloss'] = devopenloss.min()
            system_metrics[system][nstep]['min_test_openloss'] = testopenloss.min()
            system_metrics[system][nstep]['min_test_nsteploss'] = testnsteploss.min()
            for type in ssm_type:
                system_metrics[system][nstep][type] = {}
                res_system_N_type = res_system_N.loc[res_system_N.ssm_type == type]
                if not res_system_N_type.empty:
                    if res_system_N_type['metrics.open_test_loss'].idxmin() is not np.nan:
                        best = res_system_N_type.loc[res_system_N_type['metrics.open_test_loss'].idxmin()]
                    else:
                        best = None
                    res_system_N_type = res_system_N_type.loc[res_system_N_type['metrics.open_dev_loss'].notnull()]
                    devopenloss = res_system_N_type['metrics.open_dev_loss']
                    testnsteploss = res_system_N_type['metrics.nstep_test_loss']
                    testopenloss = res_system_N_type['metrics.open_test_loss']
                    system_metrics[system][nstep][type]['best'] = best
                    system_metrics[system][nstep][type]['mean_dev_openloss'] = devopenloss.mean()
                    system_metrics[system][nstep][type]['mean_test_openloss'] = testopenloss.mean()
                    system_metrics[system][nstep][type]['mean_test_nsteploss'] = testnsteploss.mean()
                    system_metrics[system][nstep][type]['std_dev_openloss'] = devopenloss.std()
                    system_metrics[system][nstep][type]['std_test_openloss'] = testopenloss.std()
                    system_metrics[system][nstep][type]['std_test_nsteploss'] = testnsteploss.std()
                    system_metrics[system][nstep][type]['min_dev_openloss'] = devopenloss.min()
                    system_metrics[system][nstep][type]['min_test_openloss'] = testopenloss.min()
                    system_metrics[system][nstep][type]['min_test_nsteploss'] = testnsteploss.min()
                    if type == 'BlockSSM':
                        for linear in linear_map:
                            system_metrics[system][nstep][type][linear] = {}
                            res_system_N_type_lin = res_system_N_type.loc[res_system_N_type.linear_map == linear]
                            if not res_system_N_type_lin.empty:
                                if res_system_N_type_lin['metrics.open_test_loss'].idxmin() is not np.nan:
                                    best = res_system_N_type_lin.loc[res_system_N_type_lin['metrics.open_test_loss'].idxmin()]
                                else:
                                    best = None
                                res_system_N_type_lin = res_system_N_type_lin.loc[res_system_N_type_lin['metrics.open_dev_loss'].notnull()]
                                devopenloss = res_system_N_type_lin['metrics.open_dev_loss']
                                testnsteploss = res_system_N_type_lin['metrics.nstep_test_loss']
                                testopenloss = res_system_N_type_lin['metrics.open_test_loss']
                                system_metrics[system][nstep][type][linear]['best'] = best
                                system_metrics[system][nstep][type][linear]['mean_dev_openloss'] = devopenloss.mean()
                                system_metrics[system][nstep][type][linear]['mean_test_openloss'] = testopenloss.mean()
                                system_metrics[system][nstep][type][linear]['mean_test_nsteploss'] = testnsteploss.mean()
                                system_metrics[system][nstep][type][linear]['std_dev_openloss'] = devopenloss.std()
                                system_metrics[system][nstep][type][linear]['std_test_openloss'] = testopenloss.std()
                                system_metrics[system][nstep][type][linear]['std_test_nsteploss'] = testnsteploss.std()
                                system_metrics[system][nstep][type][linear]['min_dev_openloss'] = devopenloss.min()
                                system_metrics[system][nstep][type][linear]['min_test_openloss'] = testopenloss.min()
                                system_metrics[system][nstep][type][linear]['min_test_nsteploss'] = testnsteploss.min()
                            for nonlinear in nonlinear_map:
                                system_metrics[system][nstep][type][linear][nonlinear] = {}
                                res_system_N_type_nonlin = res_system_N_type_lin.loc[res_system_N_type_lin.nonlinear_map == nonlinear]
                                if not res_system_N_type_nonlin.empty:
                                    if res_system_N_type_nonlin['metrics.open_test_loss'].idxmin() is not np.nan:
                                        best = res_system_N_type_nonlin.loc[res_system_N_type_nonlin['metrics.open_test_loss'].idxmin()]
                                    else:
                                        best = None
                                    res_system_N_type_nonlin = res_system_N_type_nonlin.loc[res_system_N_type_nonlin['metrics.open_dev_loss'].notnull()]
                                    devopenloss = res_system_N_type_nonlin['metrics.open_dev_loss']
                                    testnsteploss = res_system_N_type_nonlin['metrics.nstep_test_loss']
                                    testopenloss = res_system_N_type_nonlin['metrics.open_test_loss']
                                    system_metrics[system][nstep][type][linear][nonlinear]['best'] = best
                                    system_metrics[system][nstep][type][linear][nonlinear]['mean_dev_openloss'] = devopenloss.mean()
                                    system_metrics[system][nstep][type][linear][nonlinear]['mean_test_openloss'] = testopenloss.mean()
                                    system_metrics[system][nstep][type][linear][nonlinear]['mean_test_nsteploss'] = testnsteploss.mean()
                                    system_metrics[system][nstep][type][linear][nonlinear]['std_dev_openloss'] = devopenloss.std()
                                    system_metrics[system][nstep][type][linear][nonlinear]['std_test_openloss'] = testopenloss.std()
                                    system_metrics[system][nstep][type][linear][nonlinear]['std_test_nsteploss'] = testnsteploss.std()
                                    system_metrics[system][nstep][type][linear][nonlinear]['min_dev_openloss'] = devopenloss.min()
                                    system_metrics[system][nstep][type][linear][nonlinear]['min_test_openloss'] = testopenloss.min()
                                    system_metrics[system][nstep][type][linear][nonlinear]['min_test_nsteploss'] = testnsteploss.min()

metrics_df = pandas.DataFrame.from_dict(system_metrics)
metrics_df.loc['8', 'aero']['BlockSSM'].keys()
metrics_df.loc['8', 'aero']['BlockSSM']['best']
metrics_df.loc['8', 'aero']['BlockSSM']['linear']
metrics_df.loc['8', 'aero']['BlockSSM']['pf']['mlp']

# # # # # # # # # #
# METRICS
# # # # # # # # # #
idx = []
for linear in linear_map:
    for nonlinear in nonlinear_map:
        idx.append('Block_'+linear+'_'+nonlinear)
idx.append('Black')
stdopen, stdnstep, meanopen, meannstep, minopen, minnstep = \
    [pandas.DataFrame(index=idx,
      columns=N) for i in range(6)]

# choose system
system = systems[2]

# TODO: select best model based on devtest
# Block
for i in N:
    for linear in linear_map:
        for nonlinear in nonlinear_map:
            stdopen.loc['Block_'+linear+'_'+nonlinear,i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['std_test_openloss']
            stdnstep.loc['Block_'+linear+'_'+nonlinear,i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['std_test_nsteploss']
            meanopen.loc['Block_' + linear + '_' + nonlinear, i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['mean_test_openloss']
            meannstep.loc['Block_' + linear + '_' + nonlinear, i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['mean_test_nsteploss']
            minopen.loc['Block_' + linear + '_' + nonlinear, i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['min_test_openloss']
            minnstep.loc['Block_' + linear + '_' + nonlinear, i] = \
                metrics_df.loc[i, system]['BlockSSM'][linear][nonlinear]['min_test_nsteploss']
# Black
for i in N:
    stdopen.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['std_test_openloss']
    stdnstep.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['std_test_nsteploss']
    meanopen.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['mean_test_openloss']
    meannstep.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['mean_test_nsteploss']
    minopen.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['min_test_openloss']
    minnstep.loc['Black', i] = \
        metrics_df.loc[i, system]['BlackSSM']['min_test_nsteploss']

#  best model
best_model = metrics_df.loc[:, system].best

# # # # # # # # # #
# PLOTS and Tables
# # # # # # # # # #

# Latex Table
for k in [stdopen, stdnstep, meanopen, meannstep, minopen, minnstep]:
    print(k.to_latex(float_format=lambda x: '%.3f' % x))

# Bar plot
fig = plt.figure(figsize=(14, 10))
width = 0.10
ind = np.arange(len(N))
for i, n in enumerate(idx):
    plt.bar(ind+i*width, minopen.loc[n], width, label=n, edgecolor='white')
plt.xlabel('Training Prediction Horizon')
plt.ylabel('Open loop MSE')
plt.xticks(ind + width, ('2', '4', '8', '16', '32'))
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
# plt.savefig('./figs/open_loop_min_mse.eps')
# plt.savefig('./figs/open_loop_min_mse.png')

fig = plt.figure(figsize=(14, 10))
width = 0.10
ind = np.arange(len(N))
for i, n in enumerate(idx):
    plt.bar(ind+i*width, meanopen.loc[n], width, label=n, edgecolor='white', yerr=[(0, 0, 0, 0, 0), stdopen.loc[n]])
plt.xlabel('Training Prediction Horizon')
plt.ylabel('Open loop MSE')
plt.xticks(ind + .5*width,  ('2', '4', '8', '16', '32'))
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
# plt.savefig('./figs/open_loop_mean_mse.eps')
# plt.savefig('./figs/open_loop_mean_mse.png')


fig = plt.figure(figsize=(14, 10))
width = 0.10
ind = np.arange(len(N))
for i, n in enumerate(idx):
    plt.bar(ind+i*width, minnstep.loc[n], width, label=n, edgecolor='white')
plt.xlabel('Training Prediction Horizon')
plt.ylabel('N-step MSE')
plt.xticks(ind + 1.5*width,  ('2', '4', '8', '16', '32'))
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
# plt.savefig('./figs/nstep_mse.eps')
# plt.savefig('./figs/nstep_mse.png')


markers = ['x', '*', '+', 'o', 'd', 'h']
lines = ['--', '--', '--', '--', '--', '--',]
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
for n, m, l in zip(idx, markers, lines):
    ax.plot(np.arange(len(N)), minnstep.loc[n], label=n, marker=m, linestyle=l)
plt.xlabel('Training Prediction Horizon')
plt.ylabel('N-step MSE')
plt.xticks(range(len(N)),  ('2', '4', '8', '16', '32'))
plt.legend(loc='center left', bbox_to_anchor=(0, .35))
# axins = zoomed_inset_axes(ax, 2.6, loc=2) # zoom-factor: 2.5, location: upper-left
# for n, m, l in zip(idx, markers, lines):
#     axins.plot(np.arange(len(N)), minnstep.loc[n], label=n, marker=m, linestyle=l)
# plt.xticks(range(len(N)), ('2', '4', '8', '16', '32'))
# x1, x2, y1, y2 = 3.85, 5.05, 0.00, 20.05 # specify the limits
# axins.set_xlim(x1, x2) # apply the x-limits
# axins.set_ylim(y1, y2) # apply the y-limits
# ax.yaxis.tick_right()
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.tight_layout()
# plt.savefig('./figs/nstep_mse_line.eps')
# plt.savefig('./figs/nstep_mse_line.png')