import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.psl.gym import BuildingEnv
from neuromancer.plot import pltCL
import neuromancer.psl as psl

class DPCTrainer:
    def __init__(self, env, hsizes=[32, 32], lr=0.001, batch_size=100, epochs=200, xlim=(18, 22), nssm=None):
        self.env = env
        self.lr = lr
        self.hsizes = hsizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.xmin, self.xmax = xlim

        sys = self.env.model

        # Extract system parameters
        umin = torch.tensor(sys.umin)
        umax = torch.tensor(sys.umax)
        nx = sys.nx
        nu = sys.nu
        nd_obsv = sys.nd
        ny = sys.ny
        nref = sys.ny

        # Define state-space model
        state_model = Node(sys, ['x', 'u', 'd'], ['x', 'y'], name='SSM')

        # Partially observable disturbance model
        dist_model = lambda d: d[:, sys.d_idx]
        dist_obsv = Node(dist_model, ['d'], ['d_obsv'], name='dist_obsv')
        
        insize = ny + 2*nref + nd_obsv
        invars = ['y', 'ymin', 'ymax', 'd_obsv']
        
        # Augment input features with NSSM estimation
        if nssm is not None:
            nssm.eval()
            nssm = Node(nssm, ['x', 'u', 'd'], ['xh'], name='NSSM')
            insize += nx
            invars.append('xh')

        # Neural net control policy
        net = blocks.MLP_bounds(insize=insize,
                                outsize=nu, hsizes=self.hsizes,
                                nonlin=activations['gelu'],
                                min=umin, max=umax)
        self.policy = net
        policy = Node(net, invars, ['u'], name='policy')

        # Closed-loop system model
        if nssm is not None:
            nodes = [dist_obsv, policy, nssm, state_model]
        else:
            nodes = [dist_obsv, policy, state_model]
        self.cl_system = System(nodes,
                                nsteps=100,
                                name='cl_system')

        # Define objectives and constraints
        y = variable('y')
        u = variable('u')
        ymin = variable('ymin')
        ymax = variable('ymax')

        action_loss = 0.01 * (u == 0.0)
        du_loss = 0.1 * (u[:, :-1, :] - u[:, 1:, :] == 0.0)
        state_lower_bound_penalty = 50. * (y > ymin)
        state_upper_bound_penalty = 50. * (y < ymax)

        objectives = [action_loss, du_loss]
        constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

        # Create optimization problem
        self.loss = PenaltyLoss(objectives, constraints)
        self.problem = Problem([self.cl_system], self.loss)

        # Set up optimizer
        self.optimizer = torch.optim.AdamW(self.problem.parameters(), lr=self.lr)

    def get_simulation_data(self, nsim, nsteps, n_samples):
        sys = self.env.model
        nx = sys.nx
        nref = sys.ny
        y_idx = 3
        x_min, x_max = self.xmin, self.xmax
    
        #  sampled references for training the policy
        list_xmin = [x_min+(x_max-x_min)*torch.rand(1, 1)*torch.ones(nsteps+1, nref)
                    for k in range(n_samples)]
        xmin = torch.cat(list_xmin)
        batched_xmin = xmin.reshape([n_samples, nsteps+1, nref])
        batched_xmax = batched_xmin+2.0
        # get sampled disturbance trajectories from the simulation model
        list_dist = [torch.as_tensor(sys.get_D(nsteps)) for k in range(n_samples)]
        batched_dist = torch.stack(list_dist, dim=0)
        # get sampled initial conditions
        list_x0 = [torch.as_tensor(sys.get_x0().reshape(1, nx)) for k in range(n_samples)]
        batched_x0 = torch.stack(list_x0, dim=0)
        # Training dataset
        train_data = DictDataset({'x': batched_x0,
                                'y': batched_x0[:, :, [y_idx]],
                                'ymin': batched_xmin,
                                'ymax': batched_xmax,
                                'd': batched_dist},
                                name='train')

        # references for dev set
        list_xmin = [x_min+(x_max-x_min)*torch.rand(1, 1)*torch.ones(nsteps+1, nref)
                    for k in range(n_samples)]
        xmin = torch.cat(list_xmin)
        batched_xmin = xmin.reshape([n_samples, nsteps+1, nref])
        batched_xmax = batched_xmin+2.0
        # get sampled disturbance trajectories from the simulation model
        list_dist = [torch.as_tensor(sys.get_D(nsteps)) for k in range(n_samples)]
        batched_dist = torch.stack(list_dist, dim=0)
        # get sampled initial conditions
        list_x0 = [torch.as_tensor(sys.get_x0().reshape(1, nx)) for k in range(n_samples)]
        batched_x0 = torch.stack(list_x0, dim=0)
        # Development dataset
        dev_data = DictDataset({'x': batched_x0,
                                'y': batched_x0[:, :, [y_idx]],
                                'ymin': batched_xmin,
                                'ymax': batched_xmax,
                                'd': batched_dist},
                                name='dev')

        # torch dataloaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                collate_fn=train_data.collate_fn,
                                                shuffle=False)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=self.batch_size,
                                                collate_fn=dev_data.collate_fn,
                                                shuffle=False)
        return train_loader, dev_loader

    def train(self, nsim=8000, nsteps=100, nsamples=1000):
        train_loader, dev_loader = self.get_simulation_data(nsim, nsteps, nsamples)
        
        trainer = Trainer(
            self.problem,
            train_loader, dev_loader,
            optimizer=self.optimizer,
            epochs=self.epochs,
            train_metric='train_loss',
            eval_metric='dev_loss',
            warmup=self.epochs,
        )
        
        # Train control policy
        best_model = trainer.train()
        
        # load best trained model
        trainer.model.load_state_dict(best_model)
        
        return trainer.model

    def test(self, nsteps_test=2000):
        sys = self.env.model
        umin = torch.tensor(sys.umin)
        umax = torch.tensor(sys.umax)
        nx = sys.nx
        nu = sys.nu
        ny = sys.ny
        nd = sys.nd
        nref = sys.ny
        x_min, x_max = self.xmin, self.xmax
        y_idx = 3

        # generate reference
        np_refs = psl.signals.step(nsteps_test+1, 1, min=x_min, max=x_max, randsteps=5)
        ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, -1, 1)
        ymax_val = ymin_val+2.0
        # generate disturbance signal
        torch_dist = torch.tensor(sys.get_D(nsteps_test+1)).unsqueeze(0)
        # initial data for closed loop simulation
        x0 = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
        data = {'x': x0,
                'y': x0[:, :, [y_idx]],
                'ymin': ymin_val,
                'ymax': ymax_val,
                'd': torch_dist}
        self.cl_system.nsteps = nsteps_test
        # perform closed-loop simulation
        trajectories = self.cl_system(data)

        # constraints bounds
        Umin = umin * np.ones([nsteps_test, nu])
        Umax = umax * np.ones([nsteps_test, nu])
        Ymin = trajectories['ymin'].detach().reshape(-1, nref)
        Ymax = trajectories['ymax'].detach().reshape(-1, nref)
        # plot closed loop trajectories
        pltCL(Y=trajectories['y'].detach().reshape(-1, ny),
            R=Ymax,
            X=trajectories['x'].detach().reshape(-1, nx),
            D=trajectories['d'].detach().reshape(-1, nd),
            U=trajectories['u'].detach().reshape(nsteps_test, nu),
            Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)



if __name__ == '__main__':
    env = BuildingEnv(simulator='SimpleSingleZone', backend='torch')
    trainer = DPCTrainer(env, batch_size=100, epochs=10, nssm=None)
    best_model = trainer.train(nsim=100, nsteps=100, nsamples=100)
    trajectories = trainer.test(nsteps_test=2000)