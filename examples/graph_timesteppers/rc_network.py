"""
This example uses an RC graph timestepper to learn an RC Network of the
heat flow within a 5 room house. 

5 Room Layout
 ___________ 
|  0  |  1  |
|_____|_____|
| 2 | 3 | 4 |
|___|___|___|
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import neuromancer.psl as psl
from neuromancer.psl.coupled_systems import RC_Network

from neuromancer import arg
from neuromancer.dynamics import RCTimestepper
from neuromancer.dataset import SequenceDataset
from torch.utils.data import DataLoader
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
import neuromancer.psl as psl

def arg_rc(prefix=''):
    """
    Command line parser for the example.
    
    :param prefix: (str) Optional prefix for command line arguments to 
                    resolve naming conflicts.
    :return: (arg.ArgParse) Command line parser                
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("system_id")
    gp.add("-data_seed", type=int, default=408,
        help="Random seed used for simulated data")
    gp.add("-gpu", type=int, help="GPU to use")
    
    #Simulation Values
    gp.add("-nsim", type=int, default=10000,
           help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
                "train, dev, and test will be split evenly from contiguous, sequential, "
                "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
                "next nsim/3 are dev and next nsim/3 simulation steps are test points."
                "None will use a default nsim from the selected dataset or emulator")
    
    #Training Values
    gp.add("-nsteps", type=int, default=1)
    gp.add("-epochs", type=int, default=100)
    gp.add("-patience", type=int, default=100)
    gp.add("-warmup", type=int, default=0)
    
    #Model Parameters
    gp.add("-message_passing_steps", type=int, default=1)
    
    #LOG
    gp.add("-savedir", type=str, default="rc_network",
           help="Where should your trained model and plots be saved (temp)")
    return parser

def get_datasets(args):
    nsim = args.nsim
    network = RC_Network.make_5_room(nsim=nsim)
    sim = network.simulate(show_progress=True)
    x, y = sim['X'], sim['Y']
    U = network.U
    stop = int(nsim/3)

    train = {'X' : x[:stop], 'Y' : y[:stop], 'U': U[:stop]}
    dev = {'X' : x[stop:2*stop], 'Y' : y[stop:2*stop], 'U': U[stop:2*stop]}
    test = {'X' : x[2*stop:-1], 'Y' : y[2*stop:-1], 'U': U[2*stop:]}
    return network, train, dev, test
    
def get_dataloader(data, nsteps, dataname='train'):
    """
    Shapes data in dict
    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary
    :param nsteps: (int) length of windowed subsequences for N-step training.
    :param dataname: (str) Name for dataset
    :param normalizers: (dict {str: sklear.Scaler})

    :return:
    """
    dataset = SequenceDataset(data, nsteps=nsteps, name=dataname)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True,
                        collate_fn=dataset.collate_fn)
    return loader

class fu(torch.nn.Module):
    def __init__(self, nx) -> None:
        super().__init__()
        self.ext_linear = torch.nn.Parameter(
                torch.Tensor(size=(1, nx, 1)).uniform_(0.001, 0.01)
            )
        
        self.int_linear = torch.nn.Parameter(
                torch.Tensor(size=(1, nx, 1)).uniform_(0.001, 0.01)
            )
        self.in_features = nx+1
        
    def forward(self, G, U):
        U_exp, U_inp = U[:, :1], U[:, 1:]
        out1 = (U_exp[:,:,None] - G.node_attr) * self.ext_linear
        out2 = (U_inp[:,:,None] - G.node_attr) * self.int_linear
        return out1+out2

if __name__ == "__main__":
    parser = arg.ArgParser(parents=[arg_rc()])
    args, grps = parser.parse_arg_groups()
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)

    #Simulate Data and split into train/dev/test sets
    adj = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T   
    nx = 5
    modelSystem, train, dev, test = get_datasets(args)
    train_loader = get_dataloader(train, args.nsteps, 'train')
    dev_loader= get_dataloader(dev, args.nsteps, 'dev')
    test_loader = get_dataloader(test, args.nsteps, 'test')
    
    #Set up RC Graph Timestepper Model
    model = RCTimestepper(torch.from_numpy(adj), 
                        node_dim = 1,
                        message_passing_steps = args.message_passing_steps,
                        fu = fu(nx),
                        integrator='Euler',
                        input_key_map = {'x0' : 'Xp'})
    components = [model]

    #Set up loss
    yhat = variable(f"Y_pred_{model.name}")
    y = variable("Yf")
    objectives = [((yhat == y)) ^2]
    loss = PenaltyLoss(objectives,[])
    problem = Problem(components, loss)
    problem.to(device)

    #Instantiate Trainer
    optimizer = torch.optim.AdamW(problem.parameters())
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
        device=device,
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        eval_metric="nstep_dev_objective_loss",
    )
    best_model = trainer.train()
    best_output = trainer.test(best_model)
    
    """
    Rollout
    """
    rollout_pack = {'Xp':torch.tensor(test['X'][None], dtype=torch.float), 
                    'Yf': torch.tensor(test['Y'][None], dtype=torch.float), 
                    'Uf':torch.tensor(test['U'][None], dtype=torch.float)}
    out=model(rollout_pack)
    y_true = test['Y']
    y_pred=out[f'Y_pred_{model.name}'][0].detach().numpy()
    psl.plot.pltOL(Y=y_true[:1000], Ytrain=y_pred[:1000])
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    plt.savefig(os.path.join(args.savedir,"rollout.png"))
    plt.show()
