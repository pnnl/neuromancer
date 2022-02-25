import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=1)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='dlt_shared')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='dadaist')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='neuromancer')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/constrained_optim/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=2)

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
os.system('mkdir temp')

inner_loops = [1, 4, 16, 64]
etas = [1, 0.999, 0.99, 0.9]  # Expected proportion of reduction in constraints violation. Should be <= 1.
sigmas = [1.001, 1.01, 1.1, 2.0]  # Scaling factor for adaptive mu value. Shoud be > 1.0
mu_inits = [0.001, 0.01, 0.1, 1.0]   # Initial weight on constraint violations for optimization
mu_maxs = [100., 10000., 100000., 1000000.]
lrs = [0.1, 0.01, 0.001, 0.0001]

for lr in lrs:
    for inner_loop in inner_loops:
        for sigma in sigmas:
            for mu_init in mu_inits:
                for mu_max in mu_maxs:
                    for eta in etas:
                        for i in range(args.nsamples):  # 10 samples for each configuration
                            cmd = 'python mpQP_nm_2.py ' +\
                                  '-gpu 0 ' + \
                                  '-lr %s ' % lr + \
                                  '-epochs 10000 ' + \
                                  '-location %s ' % args.results + \
                                  '-system_data %s ' % 'datafile' + \
                                  '-optimizer %s ' % 'augmented_lagrange' + \
                                  '-inner_loop %s ' % inner_loop + \
                                  '-sigma %s ' % sigma + \
                                  '-mu_init %s ' % mu_init + \
                                  '-mu_max %s ' % mu_max + \
                                  '-eta %s ' % eta + \
                                  '-logger mlflow ' + \
                                  '-exp mpQP_%s_%s_%s_%s_%s_%s ' % (inner_loop, sigma, mu_init, mu_max, eta, lr) + \
                                  '-savedir temp/mpQP_%s_%s_%s_%s_%s_%s_%s ' % (inner_loop, sigma, mu_init, mu_max, eta, lr, i)
                            with open(os.path.join(args.exp_folder, 'exp_mpQP_%s_%s_%s_%s_%s_%s_%s.slurm' % (inner_loop, sigma, mu_init, mu_max, eta, lr, i)), 'w') as cmdfile: # unique name for sbatch script
                                    cmdfile.write(template + cmd)
                            sys.exit()
