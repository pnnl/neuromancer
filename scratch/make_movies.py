import os
import glob

for model in glob.glob('models/*')

for l in linmaps:
    os.system(f'python train.py -make_movie -linear_map {l} -epochs 10000 -nonlinear_map residual_mlp -state_estimator rnn')