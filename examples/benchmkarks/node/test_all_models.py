import os

for model in ['node', 'linear', 'ssm']:
    print(model)
    os.system(f'python train.py -model {model} -system Actuator -epochs 2 -iterations 5 -logdir {model}')

for inn in ['state_inclusive', 'inn', 'auto_encoder']:
    print('koopman', inn)
    os.system(f'python train.py -model koopman -inn {inn} -system Actuator -epochs 2 -iterations 5 -logdir koopman_{inn}')