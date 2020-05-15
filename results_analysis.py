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

#  add new columns placeholderds for values
res['best'] = np.nan
res['mean_dev_openloss'] = np.nan
res['mean_test_openloss'] = np.nan
res['mean_test_nsteploss'] = np.nan
res['std_dev_openloss'] = np.nan
res['std_test_openloss'] = np.nan
res['std_test_nsteploss'] = np.nan
res['min_dev_openloss'] = np.nan
res['min_test_openloss'] = np.nan
res['min_test_nsteploss'] = np.nan

# select best models
system_metrics = {}
# system_N_res = pandas.DataFrame(index=[N],columns=['devloss', 'stdopen', 'stdnstep',
#         'meanopen', 'meannstep', 'minopen', 'minnstep', 'bestrun'])

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

    # # TODO: drop this?
    # res.loc[res.datafile == system, 'best'] = best
    # res.loc[res.datafile == system, 'mean_dev_openloss'] = devopenloss.mean()
    # res.loc[res.datafile == system, 'mean_test_openloss'] = testopenloss.mean()
    # res.loc[res.datafile == system, 'mean_test_nsteploss'] = testnsteploss.mean()
    # res.loc[res.datafile == system, 'std_dev_openloss'] = devopenloss.std()
    # res.loc[res.datafile == system, 'std_test_openloss'] = testopenloss.std()
    # res.loc[res.datafile == system, 'std_test_nsteploss'] = testnsteploss.std()
    # res.loc[res.datafile == system, 'min_dev_openloss'] = devopenloss.min()
    # res.loc[res.datafile == system, 'min_test_openloss'] = testopenloss.min()
    # res.loc[res.datafile == system, 'min_test_nsteploss'] = testnsteploss.min()

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
        res_system_N =res_system.loc[res_system.nsteps == nstep]
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
                            system_metrics[system][nstep][type]['nlin_'+nonlinear] = {}
                            res_system_N_type_nonlin = res_system_N_type.loc[res_system_N_type.nonlinear_map == nonlinear]
                            if not res_system_N_type_nonlin.empty:
                                if res_system_N_type_nonlin['metrics.open_test_loss'].idxmin() is not np.nan:
                                    best = res_system_N_type_nonlin.loc[res_system_N_type_nonlin['metrics.open_test_loss'].idxmin()]
                                else:
                                    best = None
                                res_system_N_type_lin = res_system_N_type_nonlin.loc[res_system_N_type_nonlin['metrics.open_dev_loss'].notnull()]
                                devopenloss = res_system_N_type_nonlin['metrics.open_dev_loss']
                                testnsteploss = res_system_N_type_nonlin['metrics.nstep_test_loss']
                                testopenloss = res_system_N_type_nonlin['metrics.open_test_loss']
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['best'] = best
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['mean_dev_openloss'] = devopenloss.mean()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['mean_test_openloss'] = testopenloss.mean()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['mean_test_nsteploss'] = testnsteploss.mean()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['std_dev_openloss'] = devopenloss.std()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['std_test_openloss'] = testopenloss.std()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['std_test_nsteploss'] = testnsteploss.std()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['min_dev_openloss'] = devopenloss.min()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['min_test_openloss'] = testopenloss.min()
                                system_metrics[system][nstep][type]['nlin_'+nonlinear]['min_test_nsteploss'] = testnsteploss.min()

metrics_df = pandas.DataFrame.from_dict(system_metrics)
metrics_df.loc['8','aero']['BlockSSM'].keys()
metrics_df.loc['8','aero']['BlockSSM']['best']
metrics_df.loc['8','aero']['BlockSSM']['linear']
metrics_df.loc['8','aero']['BlockSSM']['pf']