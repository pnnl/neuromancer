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
from neuromancer.psl.gym import BuildingEnv


class SSM(nn.Module):
    """
    Baseline class for (neural) state space model (SSM)
    Implements discrete-time dynamical system:
        x_k+1 = fx(x_k) + fu(u_k) + fd(d_k)
    with variables:
        x_k - states
        u_k - control inputs
    """
    def __init__(self, fx, fu, fd, nx, nu, nd):
        super().__init__()
        self.fx, self.fu, self.fd = fx, fu, fd
        self.nx, self.nu, self.nd = nx, nu, nd
        self.in_features, self.out_features = nx+nu+nd, nx

    def forward(self, x, u, d):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nu])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        # state space model
        x = self.fx(x) + self.fu(u) + self.fd(d)
        return x


class NSSMTrainer:
    def __init__(self, env, hsizes=[64, 64], lr=1e-3, batch_size=100, epochs=1000):
        self.env = env
        self.lr = lr
        self.hsizes = hsizes
        self.batch_size = batch_size
        self.epochs = epochs
        
        dl = self.get_simulation_data(1, 1, self.env.model.ts)
        ny = dl.dataset[0]['Y'].shape[-1]
        nu = dl.dataset[0]['U'].shape[-1]
        nd = dl.dataset[0]['D'].shape[-1]

        fx = blocks.MLP(ny, ny, bias=True, linear_map=torch.nn.Linear, 
                        nonlin=torch.nn.ReLU, hsizes=self.hsizes)
        fu = blocks.MLP(nu, ny, bias=True, linear_map=torch.nn.Linear,
                        nonlin=torch.nn.ReLU, hsizes=self.hsizes)
        fd = blocks.MLP(nd, ny, bias=True, linear_map=torch.nn.Linear,
                        nonlin=torch.nn.ReLU, hsizes=self.hsizes)

        ssm = SSM(fx, fu, fd, ny, nu, nd)
        self.model = Node(ssm, ['yn', 'U', 'D'], ['yn'], name='NSSM')

        y = variable("Y")
        yhat = variable('yn')[:, :-1, :]

        reference_loss = 10.*(yhat == y)^2
        reference_loss.name = "ref_loss"

        onestep_loss = 1.*(yhat[:, 1, :] == y[:, 1, :])^2
        onestep_loss.name = "onestep_loss"

        objectives = [reference_loss, onestep_loss]
        constraints = []
        self.loss = PenaltyLoss(objectives, constraints)
        
    def normalize(self, x, mean, std):
        return (x - mean) / std
    
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
    
    def train(self, nsim=2000, nsteps=2, niters=5):
        ts = self.env.model.ts
        train_loader, dev_loader, test_loader = [
            self.get_simulation_data(nsim=nsim, nsteps=nsteps, ts=ts, name=name) 
            for name in ['train', 'dev', 'test']
        ]

        dynamics = System([self.model], name='system')
        problem = Problem([dynamics], self.loss)
        optimizer = torch.optim.Adam(problem.parameters(), lr=self.lr)

        trainer = Trainer(problem, train_loader, dev_loader, test_loader, optimizer, 
                          patience=100, warmup=100, epochs=self.epochs)

        for i in range(niters):  # curriculum learning
            print(f'Training with nsteps={nsteps}')
            best_model = trainer.train()
            print({k: float(v) for k, v in trainer.test(best_model).items() if 'loss' in k})
            if i == niters - 1:
                break
            nsteps *= 2
            trainer.train_data, trainer.dev_data, trainer.test_data = [
                self.get_simulation_data(nsim=nsim, nsteps=nsteps, ts=ts, name=name) 
                for name in ['train', 'dev', 'test']
            ]
            trainer.badcount = 0
            
        return best_model
            
    
if __name__ == '__main__':
    env = BuildingEnv(simulator='SimpleSingleZone')
    trainer = NSSMTrainer(env, batch_size=100, epochs=10)
    dynamics_model = trainer.train(nsim=2000, nsteps=2)