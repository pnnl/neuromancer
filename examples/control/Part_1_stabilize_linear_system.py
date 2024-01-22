import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase


if __name__ == '__main__':
    # Double integrator parameters
    nx = 2
    nu = 1
    A = torch.tensor([[1.2, 1.0],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])

    # closed loop system definition
    mlp = blocks.MLP(nx, nu, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ReLU,
                     hsizes=[20, 20, 20, 20])
    policy = Node(mlp, ['X'], ['U'])

    xnext = lambda x, u: x @ A.T + u @ B.T
    double_integrator = Node(xnext, ['X', 'U'], ['X'])
    cl_system = System([policy, double_integrator], init_func=lambda x: x)
    cl_system.show()

    # Training dataset generation
    train_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                               collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                             collate_fn=dev_data.collate_fn, shuffle=False)

    # Define optimization problem
    u = variable('U')
    x = variable('X')
    action_loss = 0.0001 * (u == 0.)^2  # control penalty
    regulation_loss = 10. * (x == 0.)^2  # target position
    loss = PenaltyLoss([action_loss, regulation_loss], [])
    problem = Problem([cl_system], loss)
    problem.show()

    # Solve the optimization problem
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        optimizer,
        epochs=400,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric='dev_loss',
        warmup=400,
    )

    # Train model with prediction horizon of 2
    cl_system.nsteps = 2
    best_model = trainer.train()

    # Test best model on a system rollout
    problem.load_state_dict(best_model)
    data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
    nsteps = 30
    cl_system.nsteps = nsteps
    trajectories = cl_system(data)
    pltCL(Y=trajectories['X'].detach().reshape(nsteps + 1, 2),
          U=trajectories['U'].detach().reshape(nsteps, 1))
    pltPhase(X=trajectories['X'].detach().reshape(nsteps + 1, 2))





