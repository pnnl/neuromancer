from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from matplotlib.lines import Line2D

from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer. modules import blocks
from neuromancer.psl.plot import get_colors
import torch 
import os 
import sys
import pprint

app = Flask(__name__)
problem_components = {'Components':[]}

def plot_system(system_name, N): 
    system = psl.systems[system_name]
    modelSystem = system()
    ts = modelSystem.ts
    raw = modelSystem.simulate(nsim=N, ts=ts)

    plot_system_output = pltOL_App(Y=raw['Y'])
    return plot_system_output


def pltOL_App(Y, Ytrain=None, U=None, D=None, X=None, figname=None):
    """
    plot trained open loop dataset
    Ytrue: ground truth training signal
    Ytrain: trained model response
    """

    plot_setup = [(name, notation, array) for
                  name, notation, array in
                  zip(['Outputs', 'States', 'Inputs', 'Disturbances'],
                      ['Y', 'X', 'U', 'D'], [Y, X, U, D]) if
                  array is not None]
    img_buffer = BytesIO()
    fig, ax = plt.subplots(nrows=len(plot_setup), ncols=1, figsize=(20, 16), squeeze=False)

    custom_lines = [Line2D([0], [0], color='gray', lw=4, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=4, linestyle='--')]
    for j, (name, notation, array) in enumerate(plot_setup):
        if notation == 'Y' and Ytrain is not None:
            colors = get_colors(array.shape[1]+1)
            for k in range(array.shape[1]):
                ax[j, 0].plot(Ytrain[:, k], '--', linewidth=2, c=colors[k])
                ax[j, 0].plot(array[:, k], '-', linewidth=2, c=colors[k])
                ax[j, 0].legend(custom_lines, ['True', 'Pred'])
        else:
            ax[j, 0].plot(array, linewidth=2)
        ax[j, 0].grid(True)
        ax[j, 0].set_title(name, fontsize=14)
        ax[j, 0].set_xlabel('Time', fontsize=14)
        ax[j, 0].set_ylabel(notation, fontsize=14)
        ax[j, 0].tick_params(axis='x', labelsize=14)
        ax[j, 0].tick_params(axis='y', labelsize=14)
    #plt.tight_layout()

    # Save the plot to a BytesIO object

    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()

    return f"data:image/png;base64,{img_str}"







@app.route('/')
def index():
    return render_template('index.html', plot_url='', system_names=['VanDerPol', 'SwingEquation'])

@app.route('/plot', methods=['POST'])
def plot():

    N = int(request.form['N'])
    system_name = request.form['system_name']
    plot_system_output = plot_system(system_name, N)
    return render_template('index.html', plot_url=plot_system_output, system_names=['VanDerPol', 'SwingEquation'])



@app.route('/create_mlp_block', methods=['POST'])
def create_mlp_block():
    global problem_components
    
    block_type = request.form.get('block_type')
    hsizes = int(request.form.get('hsizes'))
    bias = bool(request.form.get('bias')) if request.form.get('bias') else None

    if block_type == 'MLP':
        # Instantiate neuromancer.blocks.MLP with user input
        mlp_block = blocks.MLP(2, 2, bias=bias,
                               linear_map=torch.nn.Linear,
                               nonlin=torch.nn.ReLU,
                               hsizes=[hsizes, hsizes, hsizes])
        
        
        problem_components['MLP'] = mlp_block
        return 'MLP Block created and stored.'
    else:
        return 'Invalid block type'

@app.route('/get_mlp_block', methods=['GET'])
def get_mlp_block():
    global mlp_block
    return str(mlp_block)


def create_default_MLP(): 
    mlp_block = blocks.MLP(2, 2, bias=True,
                               linear_map=torch.nn.Linear,
                               nonlin=torch.nn.ReLU,
                               hsizes=[80, 80, 80]) 
    return mlp_block


@app.route('/update_problem_components', methods=['POST'])
def update_problem_components():
    if request.method == 'POST':
        new_components = []
        components = request.form.getlist('components[]')
        for comp in components: 
            if comp == 'MLP': 
                new_components.append( create_default_MLP() )

        problem_components['Components'].append(new_components)
        return 'Components updated successfully'



@app.route('/main', methods=['GET'])
def main():
    # Print all global variables
    global_vars = {name: value for name, value in globals().items() if not name.startswith("__")}
    pprint.pprint(global_vars)

    
    return 'Global variables printed in the console.'

@app.route('/print_globals', methods=['GET'])
def print_globals():
    global_vars = globals()
    return render_template('print_globals.html', global_vars=global_vars)

if __name__ == '__main__':
    app.run(debug=True)
