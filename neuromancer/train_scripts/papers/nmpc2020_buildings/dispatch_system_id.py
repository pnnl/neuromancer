import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=96)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='mpc2')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/NMPC2021_building/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch/')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=1)

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
models = ['hammerstein', 'hw']
nsteps_range = [8, 32]
hidden_states = [5, 10]
bias_onoff = ['-bias', '']

Q_con_x_vals = [0.0, 1.0]
Q_dx_vals = [0.0, 1.0]
Q_con_fdu_vals = [0.0, 1.0]
Q_e_vals = [0.0, 1.0, 10.0]

os.system('mkdir temp')

for model in models:
    for bias in bias_onoff:
        for linear in linear_map:
            for nsteps in nsteps_range:
                for nx_hidden in hidden_states:
                    for Q_con_x in Q_con_x_vals:
                        for Q_dx in Q_dx_vals:
                            for Q_con_fdu in Q_con_fdu_vals:
                                for Q_e in Q_e_vals:
                                    cmd = 'python system_id.py ' +\
                                          '-gpu 0 ' + \
                                          '-epochs 2000 ' + \
                                          '-location %s ' % args.results + \
                                          '-linear_map %s ' % linear + \
                                          '-nsteps %s ' % nsteps + \
                                          '-estimator_input_window %s ' % nsteps + \
                                          '-logger mlflow ' + \
                                          '-ssm_type %s ' % model +\
                                          '-nx_hidden %s ' % nx_hidden + \
                                          '%s ' % bias + \
                                          '-Q_con_x %s ' % Q_con_x + \
                                          '-Q_dx %s ' % Q_dx + \
                                          '-Q_con_fdu %s ' % Q_con_fdu + \
                                          '-Q_e %s ' % Q_e + \
                                          '-exp %s_%s ' % (model, nsteps) + \
                                          '-savedir temp/%s_%s_%s_%s_%s ' % (model, bias, linear, nsteps, nx_hidden) # group experiments with same configuration together
                                    with open(os.path.join(args.exp_folder, 'exp_%s_%s_%s_%s_%s_%s_%s_%s_%s.slurm' % (model, bias, linear, nsteps, nx_hidden, Q_con_x, Q_dx, Q_con_fdu, Q_e)), 'w') as cmdfile: # unique name for sbatch script
                                        cmdfile.write(template + cmd)
