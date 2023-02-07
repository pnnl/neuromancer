import os
os.system('mkdir actuator_benchmark2')

for scaled in ['', '-scaled_loss', '-normalize']:
    for model in ['node', 'linear', 'ssm']:
        print(model)
        os.system(f'python train.py -model {model} -location actuator_benchmark2/mlruns -system Actuator -epochs 100 -iterations 6 -logdir {model}{scaled} {scaled}')
    for inn in ['state_inclusive', 'inn', 'auto_encoder']:
        print('koopman', inn)
        os.system(f'python train.py -model koopman -inn {inn} -location actuator_benchmark2/mlruns -system Actuator -epochs 100 -iterations 6 -logdir koopman_{inn}{scaled} {scaled}')