"""
Mixed-Integer Differentiable Predictive Control (MI-DPC)

Reference tracking of linear system with mixed-integer control inputs using differentiable control policy via DPC algorithm

differentiable mixed-integer methodology is adopted from: https://arxiv.org/abs/2506.19646
"""

import torch
from neuromancer.system import Node, SystemPreview
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import gumbel_softmax
from neuromancer.modules.activations import activations

if __name__ == "__main__":
    
    """
    # # #  Straight-through estimator (STE) method to be used
    """
    STE_methods = ['sigmoid', 'softmax']
    STE_method = STE_methods[0]

    if STE_method == 'sigmoid':
       def relaxed_round(x, slope=10.0): # differentiable nearest integer rounding via Sigmoid STE
           backward = (x-torch.floor(x)-0.5) # fractional value with rounding threshold
           return torch.round(x) + (torch.sigmoid(slope*backward) - torch.sigmoid(slope*backward).detach())
           #  forward pass↑     backward pass↑                                               no grad↑

    """
    # # # System dynamics 
    """
    a_1, a_2, _labmda_ = 0.95, 0.85, 0.0020 # model parameters
    b_1, b_2, e_1, e_2 = 0.2, 0.0825, 0.1, 0.1

    A = torch.tensor([[a_1 - _labmda_, _labmda_], # state dynamics
                    [_labmda_, a_2 - _labmda_]])
    B = torch.tensor([[0.7*b_1, 0],   # input dynamics
                     [0.3*b_1, b_2]])
    E = torch.tensor([[-e_1], [-e_2]]) # disturbance gains
    
    nx, nref = 2, 2 # number of states, number of references 
    nu, ndelta, nd = 1, 1, 1 # number of continous inputs, integer inputs, and disturbances

    dynamics = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T # plant model

    # process constraints
    x1_min, x2_min = 0.0, 0.0
    x1_max, x2_max = 8, 4
    u_delta_min, u_delta_max = 0, 5 # integer control input bounds
    u_c_min, u_c_max = 0.0, 7.0     # continous control input bounds


    """
    # # # Control policy
    """
    nsteps = 20 # prediction horizon length    
    integers = torch.arange(u_delta_min, u_delta_max+.1, dtype=torch.get_default_dtype()).unsqueeze(0) # vector of feasible integers
    
    net_continous = blocks.MLP_bounds(insize=nx+(nref+nd)*(nsteps+1), outsize=nu, hsizes=[64,64], # neural module for continous control inputs
                             nonlin=activations['gelu'], min=u_c_min, max=u_c_max)
    
    int_out_size = integers.shape[-1] if STE_method == 'softmax' else ndelta
    net_integer = blocks.MLP(insize=nx+(nref+nd)*(nsteps+1), outsize=int_out_size, hsizes=[64,64], # neural module for integer control inputs
                             nonlin=activations['gelu'])

    """
    # # # Computational graph components
    """
    dynamics_node = Node(dynamics, ['x','u','d'], ['x'], name='dynamics_model') # system dynamics

    continous_policy_node = Node(net_continous, ['x','r','d'], ['u_c'], name='continous_input_policy')
    integer_policy_node = Node(net_integer, ['x','r','d'], ['u_delta'], name='integer_input_policy')
    
    # Define soft rounding nodes
    if STE_method == 'sigmoid':
        slope = 10
        round_fn = lambda u_c, u_delta_relaxed: torch.cat((u_c, torch.clip(relaxed_round(u_delta_relaxed, slope=slope),u_delta_min,u_delta_max)), dim=-1)
    
    elif STE_method == 'softmax':
        tau = 2 # temperature coefficient in (0,inf)
        round_fn = lambda u_c, u_delta_relaxed: torch.cat((u_c, gumbel_softmax(u_delta_relaxed, tau=tau, hard=True) @ integers.T), dim=-1)
        
    rounding_node = Node(round_fn, input_keys=['u_c', 'u_delta'], output_keys=['u'], name='soft_rounding')

    """
    # # # Closed-loop system
    """
    cl_system = SystemPreview([continous_policy_node, integer_policy_node, rounding_node, dynamics_node], # computational graph
                              nsteps=nsteps, name='cl_system', pad_mode='reflect',
                              preview_keys_map={'r': ['continous_input_policy', 'integer_input_policy'],  # preview references for both control policy modules
                                                'd': ['continous_input_policy', 'integer_input_policy']} )# preview disturbance for both control policy modules

    """
    # # # Training dataset
    """
    num_data, batch_size = 14000, 2000
    # sample disturbanecs
    dist_magnitude = 5
    dist_data = np.random.beta(0.2, 1.4, (num_data*(nsteps+1)))*dist_magnitude
    dist_data_torch = torch.tensor(dist_data, dtype=torch.get_default_dtype()).reshape(num_data,nsteps+1,nd)
    # sample references
    ref1_baseline, ref2_baseline = 4, 2 # reference baseline
    ref1_data = ref1_baseline + 1.5*np.sin(0.05*np.arange(num_data*(nsteps+1))).reshape(-1,1)
    ref2_data = ref2_baseline + 0.5*np.cos(0.05*np.arange(num_data*(nsteps+1))).reshape(-1,1)
    ref_data = np.concatenate((ref1_data, ref2_data), axis=-1)
    ref_data_torch = torch.tensor(ref_data, dtype=torch.get_default_dtype()).reshape(num_data,nsteps+1,nref)
    # sample initial conditions
    x1_data = torch.empty(num_data, 1, 1).uniform_(x1_min, x1_max)
    x2_data = torch.empty(num_data, 1, 1).uniform_(x2_min, x2_max)
    x_data = torch.cat((x1_data, x2_data), dim=-1)

    num_train_data = 10000 # training data splits to train data and development data

    train_data = DictDataset({'x': x_data[:num_train_data], 'd': dist_data_torch[:num_train_data], 'r': ref_data_torch[:num_train_data]}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'x': x_data[num_train_data:], 'd': dist_data_torch[num_train_data:], 'r': ref_data_torch[num_train_data:]}, name='dev')
    # instantiate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, collate_fn=dev_data.collate_fn)
    
    """
    # # # Control objectives
    """
    u = variable('u') # control inputs
    x = variable('x') # system states
    r = variable('r') # referece
    # control objectives
    Q_control, R_smoothing = 10., 1. # reference tracking and control smoothing weights
    tracking_loss = Q_control * ((x == r)^2)     # reference tracking loss
    input_smoothing_loss = R_smoothing * ((u[:, 1:, :] == u[:, :-1, :])^2) # control smoothing loss
    tracking_loss.name, input_smoothing_loss.name = 'tracking_loss', 'smoothing_loss'
    # constraints
    x1_lb, x1_ub = 50.*(x[:,:, [0]] >= x1_min), 50.*(x[:,:, [0]] <= x1_max)
    x2_lb, x2_ub = 50.*(x[:,:, [1]] >= x2_min), 50.*(x[:,:, [1]] <= x2_max)
    u_c_lb, u_c_ub = 50.*(u[:,:, [0]] >= u_c_min), 50.*(u[:,:, [0]] <= u_c_max)
    x1_lb.name, x1_ub.name = 'x1_lb', 'x1_ub'
    x2_lb.name, x2_ub.name = 'x2_lb', 'x2_ub'
    u_c_lb.name, u_c_ub.name = 'u_lb', 'u_ub' 
    
    constraints = [x1_lb, x1_ub, x2_lb, x2_ub, u_c_lb, u_c_ub]
    loss = PenaltyLoss([tracking_loss, input_smoothing_loss], constraints)
    problem = Problem([cl_system], loss)

    optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.0003, amsgrad=False, weight_decay=0.)
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=200,
        train_metric='train_loss',
        dev_metric='dev_loss',
        eval_metric='dev_loss',
        warmup=20,
        patience=80,
    )
    best_model = trainer.train()    # start optimization
    trainer.model.load_state_dict(best_model) # load best trained model

    """
    Control policy test
    """
    s_length = 100 # number of simulation steps
    dists = np.random.beta(0.2, 1., (s_length+nsteps))*dist_magnitude # disturbances 
    dists_torch = torch.tensor(dists, requires_grad=False, dtype=torch.get_default_dtype()).reshape(1,-1,nd)
    ref1 = ref1_baseline + 1.*np.sin(0.1*np.arange(s_length+nsteps)).reshape(-1,1) # references
    ref2 = ref2_baseline + 1.*np.cos(0.1*np.arange(s_length+nsteps)).reshape(-1,1)
    refs = np.concatenate((ref1,ref2),axis=-1)
    refs_torch = torch.tensor(refs, requires_grad=False, dtype=torch.get_default_dtype()).reshape(1,-1,nref)

    sim_data = {'x': 3*torch.rand(1, 1, nx),
            'r': refs_torch,
            'd': dists_torch}
    
    cl_system.nsteps = s_length
    with torch.no_grad():
        trajectories = cl_system(sim_data) # simulate
    
    """
    Plotting
    """
    fig, ax = plt.subplots(2,2, figsize=(10,4))
    ax[0,0].axhline(x1_min, linestyle=':', c='k', label='bounds'); ax[0,0].axhline(x1_max, linestyle=':', c='k')
    ax[0,0].plot(trajectories['r'][0,:s_length,0], 'k--', label='reference'); ax[0,0].plot(trajectories['x'][0,:,0], label='process'); 
    ax[1,0].plot(trajectories['x'][0,:,1]); ax[1,0].plot(trajectories['r'][0,:s_length,1], 'k--') 
    [ax[1,0].axhline(i, linestyle=':', c='k') for i in [x2_min, x2_max]]
    [ax[0,1].axhline(i, linestyle=':', c='k') for i in [u_c_min, u_c_max]]; ax[0,1].plot(trajectories['u'][0,:,0]); 
    [ax[1,1].axhline(i, linestyle=':', c='k') for i in [u_delta_min, u_delta_max]]; 
    ax[1,1].step(np.arange(0,len(trajectories['u'][0,:,1])),trajectories['u'][0,:,1], linewidth=1.5)
   
    [ ( i.grid(), i.set_xlabel("time"), i.legend()) for i in ax.flat ]
    ax[0,0].set_ylabel('$x_1$'); ax[1,0].set_ylabel('$x_2$'); ax[0,1].set_ylabel('continous input'); ax[1,1].set_ylabel('integer input')
    plt.show()

    plt.figure(figsize=(5,2)); plt.plot(trajectories['d'][0,:s_length,0]); 
    plt.ylabel('disturbances'); plt.xlabel('time'); plt.grid()
    plt.show()