import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=96)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='shared_dlt')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='pins')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='dltneuro')
parser.add_argument('-results', type=str, help='Where to log mlflow results', default='/qfs/projects/deepmpc/mlflow/nonlin_sysid_ablation_2020_09_24/mlruns')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch/')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=5)
parser.add_argument('-params_file', default='best_systems.csv')

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

ablations = [('Q_dx', 0.0),
             ('Q_con_x', 0.0),
             ('Q_con_fdu', 0.0),
             ('linear_map', 'linear')]
params = pd.read_csv(args.params_file)
stub = 'python system_id.py -gpu 0 -epochs 15000 -location %s -logger mlflow -exp allablate -warmup 100 -patience 100 ' % args.results
for j in params.index:
    for ab in ablations:
        params.loc[j, [ab[0]]] = ab[1]

print([k for k in params.columns])
for j in params.index:
    print(j)
    for i in range(5):
        exp = params['system'][j]
        savedir = '_'.join(['noablate', params['system'][j], str(i)])
        argstring = ' '.join(['-' + p + ' ' + str(params[p][j]) for p in params.columns] + [' -savedir %s ' % savedir])
        cmd = stub + argstring
        print(cmd)
        with open(os.path.join(args.exp_folder, 'exp_%s.slurm' % savedir), 'w') as cmdfile: # unique name for sbatch script
            cmdfile.write(template + cmd)


