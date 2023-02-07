import os

for model in ['node', 'linear', 'ssm']:
    print(model)
    os.system(f'python train.py -model {model} -system Actuator -epochs 20 -iterations 5 -logdir scaled_{model} -scaled_loss')

for inn in ['state_inclusive', 'inn', 'auto_encoder']:
    print('koopman', inn)
    os.system(f'python train.py -model koopman -inn {inn} -system Actuator -epochs 20 -iterations 5 -logdir scaled_koopman_{inn} -scaled_loss')