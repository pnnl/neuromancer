import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=96)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='pins')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='mpc2')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/EED_building_exp_2020_8_5/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch/')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=5)

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

linear_map = ['linear', 'pf', 'softSVD']
nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
models = ['blocknlin', 'blackbox']
Q_values = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.2)]  # Q_con_x, Q_dx, Q_con_fdu
constrainted = ['unconstr', 'constr']
bias_onoff = ['-bias', '']
nsteps_range = [1, 8, 16, 32, 64]
os.system('mkdir ../deepmpc/temp')

for model in models:
    for bias in bias_onoff:
        for constr, Q_val in zip(constrainted, Q_values):
            for linear in linear_map:
                for nonlinear in nonlinear_map:
                    for nsteps in nsteps_range:
                        for i in range(args.nsamples):
                            cmd = 'python ../deepmpc/system_id_eed_build.py ' +\
                                  '-gpu 0 ' + \
                                  '-lr 0.003' + \
                                  '-epochs 6000 ' + \
                                  '-location %s ' % args.results + \
                                  '-linear_map %s ' % linear + \
                                  '-nonlinear_map %s ' % nonlinear + \
                                  '-nsteps %s ' % nsteps + \
                                  '-mlflow ' + \
                                  '-ssm_type %s ' % model +\
                                  '-nx_hidden 4' + \
                                  '%s ' % bias + \
                                  '-Q_con_x %s -Q_dx %s -Q_con_fdu %s ' % Q_val + \
                                  '-exp %s_%s_%s ' % (model, constr, nsteps) + \
                                  '-savedir ../deepmpc/temp/%s_%s_%s_%s_%s_%s_%s ' % (model, constr, bias, linear, nonlinear, nsteps, i) # group experiments with same configuration together - TODO: add more params
                            with open(os.path.join(args.exp_folder, 'exp_%s_%s_%s_%s_%s_%s_%s.slurm' % (model, constr, bias, linear, nonlinear, nsteps, i)), 'w') as cmdfile: # unique name for sbatch script
                                cmdfile.write(template + cmd)
