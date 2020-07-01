import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='mpc2')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/flexy_exp_2020_6_30/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch/')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=10)

args = parser.parse_args()

template = '#!/bin/bash\n' +\
           '#SBATCH -A %s\n' % args.allocation +\
           '#SBATCH -t %s:00:00\n' % args.hours +\
           '#SBATCH --gres=gpu:1\n' +\
           '#SBATCH -p %s\n' % args.partition +\
           '#SBATCH -N 1\n' +\
           '#SBATCH -n 2\n' +\
           '#SBATCH -o %j.out\n' +\
           '#SBATCH -e %j.err\n' +\
           'source /etc/profile.d/modules.sh\n' +\
           'module purge\n' +\
           'module load python/anaconda3.2019.3\n' +\
           'ulimit\n' +\
           'source activate %s\n\n' % args.env

os.system('mkdir %s' % args.exp_folder)

systems = ['flexy_air']
linear_map = ['linear', 'pf', 'softSVD']
nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
models = ['blocknlin']
Q_values = [(0.0, 0.0, 0.0, 0.0), (0.2, 0.2, 0.2, 0.2)]
constrainted = ['unconstr', 'constr']

# choose nsim: 2880 for 30 days, 5760 for 60 days, 8640 for 90 days
nsteps_range = [1, 8, 16, 32, 64]
os.system('mkdir temp')
for system in systems:
    for model in models:
        for bias in ['-bias']:
            for constr, Q_val in zip(constrainted, Q_values):
                for linear in linear_map:
                    for nonlinear in nonlinear_map:
                        for nsteps in nsteps_range:
                            for i in range(args.nsamples): # 10 samples for each configuration
                                cmd = 'python ./deepmpc/train.py ' +\
                                      '-gpu 0 ' + \
                                      '-lr 0.003 ' + \
                                      '-epochs 10000 ' + \
                                      '-location %s ' % args.results + \
                                      '-system_data %s ' % 'datafile' + \
                                      '-system %s ' % system + \
                                      '-linear_map %s ' % linear + \
                                      '-nonlinear_map %s ' % nonlinear + \
                                      '-nsteps %s ' % nsteps + \
                                      '-mlflow ' + \
                                      '-ssm_type %s ' % model +\
                                      '-nx_hidden 10 ' + \
                                      '%s ' % bias + \
                                      '-Q_con_u %s -Q_con_x %s -Q_dx_ud %s -Q_dx %s ' % Q_val + \
                                      '-exp %s_%s_%s_%s_%s_%s_%s ' % (system, model, constr, bias, linear, nonlinear, nsteps) + \
                                      '-savedir temp/%s_%s_%s_%s_%s_%s_%s_%s ' % (system, model, constr, bias, linear, nonlinear, nsteps, i) # group experiments with same configuration together - TODO: add more params
                                with open(os.path.join(args.exp_folder, 'exp_%s_%s_%s_%s_%s_%s_%s_%s.slurm' % (system, model, constr, bias, linear, nonlinear, nsteps, i)), 'w') as cmdfile: # unique name for sbatch script
                                    cmdfile.write(template + cmd)


# BlackSSM - all datasets all nonlinear maps, default linear
for system in systems:
    for bias in ['-bias']:
        for constr, Q_val in zip(constrainted, Q_values):
            for linear in linear_map:
                for nonlinear in nonlinear_map:
                    for nsteps in nsteps_range:
                        for i in range(args.nsamples): # 10 samples for each configuration
                            cmd = 'python ./deepmpc/train.py ' +\
                                  '-gpu 0 ' + \
                                  '-lr 0.003 ' + \
                                  '-epochs 10000 ' + \
                                  '-location %s ' % args.results + \
                                  '-system_data %s ' % 'datafile' + \
                                  '-system %s ' % system + \
                                  '-linear_map %s ' % linear + \
                                  '-nonlinear_map %s ' % nonlinear + \
                                  '-nsteps %s ' % nsteps + \
                                  '-mlflow ' + \
                                  '-ssm_type blackbox ' + \
                                  '-nx_hidden 10 ' +\
                                  '%s ' % bias + \
                                  '-Q_con_u %s -Q_con_x %s -Q_dx_ud %s -Q_dx %s ' % Q_val + \
                                  '-exp %s_blackbox_%s_%s_%s_%s_%s ' % (system, constr, bias, linear, nonlinear, nsteps) + \
                                  '-savedir temp/%s_blackbox_%s_%s_%s_%s_%s_%s ' % (system, constr, bias, linear, nonlinear, nsteps, i)
                            with open(os.path.join(args.exp_folder, 'exp_%s_blackbox_%s_%s_%s_%s_%s_%s.slurm' % (system, constr, bias, linear, nonlinear, nsteps, i)), 'w') as cmdfile: # unique name for sbatch script
                                    cmdfile.write(template + cmd)
