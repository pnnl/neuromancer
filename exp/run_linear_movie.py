import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='pins')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='mpc2')
parser.add_argument('-results', type=str, help='Where to log mlflow results',
                    default='/qfs/projects/deepmpc/mlflow/linear_map_comparison/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch/')
parser.add_argument('-gpu', type=int, default=0)


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

# Block SSM without bias, vehicle, basic nlin options
datapaths = ['./datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat']
systems = ['aero']
linear_map=['pf', 'linear', 'softSVD', 'sparse']
nonlinear_map= ['sparse_residual_mlp']
for linear in linear_map:
    for path, system in zip(datapaths, systems):
        for bias in ['']:
            for nonlinear in nonlinear_map:
                for nsteps in [4]:
                    for i in range(1): # 10 samples for each configuration
                        cmd = 'python train.py ' +\
                              '-gpu %s ' % args.gpu +\
                              '-epochs 30000 ' + \
                              '-location %s ' % args.results + \
                              '-datafile %s ' % path + \
                              '-linear_map %s ' % linear + \
                              '-nonlinear_map %s ' % nonlinear + \
                              '-nsteps %s ' % nsteps + \
                              '-mlflow -make_movie ' + \
                              '-ssm_type BlockSSM ' + \
                              '%s ' % bias + \
                              '-exp BlockSSM_%s_%s_%s_%s_%s ' % (system, linear, nonlinear, bias, nsteps) + \
                              '-savedir linear_map_%s_%s_%s_%s_%s_%s ' % (system, linear, nonlinear, bias, nsteps, i) # group experiments with same configuration together - TODO: add more params
                        # with open(os.path.join(args.exp_folder, 'exp_linear_map_%s_%s_%s_%s_%s_%s.slurm' % (system, linear, nonlinear, bias, nsteps, i)), 'w') as cmdfile: # unique name for sbatch script
                        os.system(cmd)

# # Block SSM without bias, vehicle, basic nlin options
# datapaths = ['xxx']
# systems = ['building']
# nonlinear_map= ['rnn']
# for linear in linear_map:
#     for path, system in zip(datapaths, systems):
#         for bias in ['']:
#             for nonlinear in nonlinear_map:
#                 for nsteps in [4]:
#                     for i in range(1): # 10 samples for each configuration
#                         cmd = 'python train_building.py ' +\
#                               '-gpu 0 ' +\
#                               '-epochs 30000 ' + \
#                               '-location %s ' % args.results + \
#                               '-datafile %s ' % path + \
#                               '-linear_map %s ' % linear + \
#                               '-nonlinear_map %s ' % nonlinear + \
#                               '-nsteps %s ' % nsteps + \
#                               '-mlflow -make_movie ' + \
#                               '-exp BlockSSM_%s_%s_%s_%s_%s ' % (system, linear, nonlinear, bias, nsteps) + \
#                               '-savedir linear_map_%s_%s_%s_%s_%s_%s ' % (system, linear, nonlinear, bias, nsteps, i) # group experiments with same configuration together - TODO: add more params
#                         with open(os.path.join(args.exp_folder, 'exp_linear_map_%s_%s_%s_%s_%s_%s.slurm' % (system, linear, nonlinear, bias, nsteps, i)), 'w') as cmdfile: # unique name for sbatch script
#                             cmdfile.write(template + cmd)
