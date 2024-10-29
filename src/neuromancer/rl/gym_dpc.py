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
import neuromancer.psl as psl
import numpy as np

class DPCTrainer:
    def __init__(self, env, hsizes=[32, 32], lr=0.001, batch_size=100, epochs=200):
        self.env = env
        self.lr = lr
        self.hsizes = hsizes
        self.batch_size = batch_size
        self.epochs = epochs

        sys = self.env.model

        # Extract system parameters
        A = torch.tensor(sys.params[2]['A'])
        B = torch.tensor(sys.params[2]['Beta'])
        C = torch.tensor(sys.params[2]['C'])
        E = torch.tensor(sys.params[2]['E'])
        umin = torch.tensor(sys.umin)
        umax = torch.tensor(sys.umax)
        nx = sys.nx
        nu = sys.nu
        nd = E.shape[1]
        nd_obsv = sys.nd
        ny = sys.ny
        nref = sys.ny
        y_idx = 3

        # Define state-space model
        xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
        state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')
        ynext = lambda x: x @ C.T
        output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

        # Partially observable disturbance model
        dist_model = lambda d: d[:, sys.d_idx]
        dist_obsv = Node(dist_model, ['d'], ['d_obsv'], name='dist_obsv')

        # Neural net control policy
        net = blocks.MLP_bounds(insize=ny + 2*nref + nd_obsv,
                                outsize=nu, hsizes=self.hsizes,
                                nonlin=activations['gelu'],
                                min=umin, max=umax)
        policy = Node(net, ['y', 'ymin', 'ymax', 'd_obsv'], ['u'], name='policy')

        # Closed-loop system model
        self.cl_system = System([dist_obsv, policy, state_model, output_model],
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

        # Set up optimizer and trainer
        self.optimizer = torch.optim.AdamW(self.problem.parameters(), lr=self.lr)
        self.trainer = Trainer(self.problem, None, None, self.optimizer, epochs=self.epochs)

    def get_simulation_data(self, nsim, nsteps, ts, name='data'):
        nsim = nsim // nsteps * nsteps
        sim = self.env.model.simulate(nsim=nsim, ts=ts)
        sim = {k: sim[k] for k in ['X', 'Y', 'U', 'D']}
        nbatches = nsim // nsteps
        for key in sim:
            m = self.env.model.stats[key]['mean']
            s = self.env.model.stats[key]['std']
            x = self.normalize(sim[key], m, s).reshape(nbatches, nsteps, -1)
            x = torch.tensor(x, dtype=torch.float32)
            sim[key] = x
        sim['yn'] = sim['Y'][:, :1, :]
        ds = DictDataset(sim, name=name)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=ds.collate_fn, shuffle=True)

    def normalize(self, x, mean, std):
        return (x - mean) / std

    def train(self, nsim=2000, nsteps=2, niters=5):
        ts = self.env.model.ts
        train_loader, dev_loader, test_loader = [
            self.get_simulation_data(nsim=nsim, nsteps=nsteps, ts=ts, name=name) 
            for name in ['train', 'dev', 'test']
        ]

        self.trainer.train_data = train_loader
        self.trainer.dev_data = dev_loader
        self.trainer.test_data = test_loader

        for i in range(niters):
            print(f'Training with nsteps={nsteps}')
            best_model = self.trainer.train()
            print({k: float(v) for k, v in self.trainer.test(best_model).items() if 'loss' in k})
            if i == niters - 1:
                break
            nsteps *= 2
            self.trainer.train_data, self.trainer.dev_data, self.trainer.test_data = [
                self.get_simulation_data(nsim=nsim, nsteps=nsteps, ts=ts, name=name) 
                for name in ['train', 'dev', 'test']
            ]
            self.trainer.badcount = 0

        return best_model

    def test(self, nsteps_test=2000):
        sys = self.env.model
        x_min = 18.
        x_max = 22.
        np_refs = psl.signals.step(nsteps_test+1, 1, min=x_min, max=x_max, randsteps=5)
        ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
        ymax_val = ymin_val + 2.0
        torch_dist = torch.tensor(sys.get_D(nsteps_test+1)).unsqueeze(0)
        x0 = torch.tensor(sys.get_x0()).reshape(1, 1, sys.nx)
        data = {'x': x0,
                'y': x0[:, :, [3]],
                'ymin': ymin_val,
                'ymax': ymax_val,
                'd': torch_dist}
        self.cl_system.nsteps = nsteps_test
        trajectories = self.cl_system(data)
        return trajectories

if __name__ == '__main__':
    env = BuildingEnv(simulator='SimpleSingleZone')
    trainer = DPCTrainer(env, batch_size=100, epochs=10)
    best_model = trainer.train(nsim=2000, nsteps=2)
    trajectories = trainer.test(nsteps_test=2000)