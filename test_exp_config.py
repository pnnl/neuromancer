import os

gpu='cpu'
epochs=2
lr=0.003,
nsteps=[1, 4, 8, 16, 32]

datapaths = ['./datasets/NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat',
                 './datasets/NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat',
                 './datasets/NLIN_MIMO_CSTR/NLIN_MIMO_CSTR2.mat',
                 './datasets/NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat']

systems = ['tank','vehicle3','reactor','aero']
ssm_type=['BlockSSM', 'BlackSSM']
nx_hidden=10
state_estimator='rnn'
linear_map=['pf', 'linear']
nonlinear_map= ['mlp', 'residual_mlp', 'sparse_residual_mlp', 'linear', 'rnn']

# Block
for path in datapaths:
    for linear in linear_map:
        for nonlinear in nonlinear_map:
            for bias in ['-bias', '']:
                os.system(f'python train.py -datafile {path} -ssm_type BlockSSM -linear_map {linear} '
                          f'-nonlinear_map {nonlinear} -state_estimator rnn {bias} -epochs {epochs}')
                print(linear+nonlinear)

# Black
for path, system in zip(datapaths, systems):
    for nonlinear in nonlinear_map:
        for bias in ['-bias', '']:
            os.system(f'python train.py -datafile {path} -ssm_type BlackSSM '
                          f'-nonlinear_map {nonlinear} -state_estimator rnn {bias} -epochs {epochs}')
            print(bias + nonlinear)




