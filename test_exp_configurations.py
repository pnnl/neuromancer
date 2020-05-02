import os


gpu='cpu'
epochs=2
lr=0.003,
nsteps=[1, 4, 8, 16, 32]
datafiles=[os.path.join('datasets', k) for k in ['NLIN_MIMO_vehicle/NLIN_MIMO_vehicle.mat',
                                                    'NLIN_SISO_two_tank/NLIN_SISO_two_tank.mat',
                                                 'NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat',
                                                 'NLIN_MIMO_CSTR/NLIN_MIMO_CSTR.mat']]
ssm_type=['BlockSSM', 'BlackSSM']
nx_hidden=40
state_estimator='rnn'
linear_map=['pf', 'spectral', 'vanilla']
bias=True

for datafile in datafiles:
    # os.system(f'python train_loop.py -datafile {datafile} -ssm_type BlackSSM')
    for linear in linear_map:
        os.system(f'python train_loop.py -datafile {datafile} -ssm_type BlockSSM -linear_map {linear} -state_estimator rnn -bias')
        break
    break






