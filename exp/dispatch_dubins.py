import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='dltneuro')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/dubins_exp_2020_9_17/mlruns')
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
datatypes = ['emulator']
systems = ['UAV3D_kin']
linear_map = ['linear', 'softSVD']
nonlinear_map = ['mlp', 'residual_mlp', 'rnn']
models = ['wiener', 'hw', 'hammerstein', 'blocknlin']
Q_con_y = [0.1, 1.0, 10.0]
sigma_mins = [0.1, 0.5, 0.9]
nsteps_range = [4, 32]
os.system('mkdir temp')
for system, datatype in zip(systems, datatypes):
    for model in models:
        for linear in linear_map:
            for nonlinear in nonlinear_map:
                for nsteps in nsteps_range:
                    for i in range(args.nsamples):
                        cmd = 'python ../neuromancer/train_scripts/system_id_nonlin_UAV.py ' +\
                              '-gpu 0 ' + \
                              '-epochs 1000 ' + \
                              '-location %s ' % args.results + \
                              '-system_data %s ' % datatype + \
                              '-system %s ' % system + \
                              '-linear_map %s ' % linear + \
                              '-nonlinear_map %s ' % nonlinear + \
                              '-nsteps %s ' % nsteps + \
                              '-estimator_input_window %s ' % nsteps + \
                              '-logger mlflow ' + \
                              '-ssm_type %s ' % model +\
                              '-Q_con_x %s ' % random.choice(Q_con_y) + \
                              '-sigma_min %s ' % random.choice(sigma_mins) + \
                              '-exp %s ' % (system) + \
                              '-savedir temp/%s_%s_%s_%s_%s_%s ' % (system, model, linear, nonlinear, nsteps, i)

                        with open(os.path.join(args.exp_folder, 'exp_%s_%s_%s_%s_%s_%s.slurm' % (system, model, linear, nonlinear, nsteps, i)), 'w') as cmdfile:  # unique name for sbatch script
                            cmdfile.write(template + cmd)