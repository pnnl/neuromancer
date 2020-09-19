import os
import argparse
import random
from system_id_timedelay_fsw import datasplits

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='dltneuro')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/fsw_second_exp_2020_09_18/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files', default='sbatch/')

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

lr = (0.0005, 0.003)
nsteps = [16, 32, 64, 128]
ssm_type = ['blackbox', 'hw', 'hammerstein', 'blocknlin', 'linear']
nx_hidden = range(5, 50, 5)
n_layers = range(1, 11)
state_estimator = ['mlp', 'linear', 'residual_mlp']
estimator_input_window = range(1, 11)
linear_map = ['linear', 'pf', 'softSVD', 'lasso']
nonlinear_map = ['mlp', 'linear', 'residual_mlp']
bias = [0, 1]
activation = ['gelu', 'relu', 'tanh', 'softexp', 'blu']
timedelay = range(11)
Q_con_x = (1.0, 10.0)
Q_dx = (0.1, 1.0)
Q_sub = (0.1, 1.0)
Q_y = (1.0, 10.0)
Q_e = (1.0, 10.0)
Q_con_fdu = (0.0, 1.0)
systems = ['fsw_phase_1', 'fsw_phase_2']
for i in range(100):
    for system in systems:
        for tset in datasplits:
            cmd = 'python system_id_timedelay_fsw.py ' + \
                  '-gpu 0 ' + \
                  '-patience 100 ' + \
                  '-warmup 100 ' + \
                  '-lr %s ' % random.uniform(*lr) + \
                  '-epochs 15000 ' + \
                  '-location %s ' % args.results + \
                  '-lr_scheduler ' + \
                  '-system %s ' % system + \
                  '-activation %s ' % random.choice(activation) + \
                  '-linear_map %s ' % random.choice(linear_map) + \
                  '-timedelay %s ' % random.choice(timedelay) + \
                  '-n_layers %s ' % random.choice(n_layers) + \
                  '-estimator_input_window %s ' % random.choice(estimator_input_window) + \
                  '-nonlinear_map %s ' % random.choice(nonlinear_map) + \
                  '-nsteps %s ' % random.choice(nsteps) + \
                  '-logger mlflow ' + \
                  '-ssm_type %s ' % random.choice(ssm_type) + \
                  '-state_estimator %s ' % random.choice(state_estimator) + \
                  '-nx_hidden %s ' % random.choice(nx_hidden) + \
                  '-bias %s ' % random.choice(bias) + \
                  '-trainset %s ' % tset +\
                  '-Q_con_fdu %s -Q_con_x %s -Q_dx %s -Q_sub %s -Q_y %s -Q_e %s ' % (random.uniform(*Q_con_fdu),
                                                                                     random.uniform(*Q_con_x),
                                                                                     random.uniform(*Q_dx),
                                                                                     random.uniform(*Q_sub),
                                                                                     random.uniform(*Q_y),
                                                                                     random.uniform(*Q_e)) + \
                  '-exp %s_%s ' % (system, tset) + \
                  '-savedir temp/%s_%s_%s ' % (system, tset, i)
            with open(os.path.join(args.exp_folder, 'exp_%s_%s_%s.slurm' % (system, tset, i)), 'w') as cmdfile: # unique name for sbatch script
                cmdfile.write(template + cmd)

