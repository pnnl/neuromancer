"""
Tutorial Script for using the graph timestepper
"""
import torch
import torch.nn as nn
import neuromancer.arg as arg
from torch.utils.data import DataLoader
from neuromancer.graph_dynamics import GraphTimestepper
from neuromancer.dataset import GraphDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.loggers import MLFlowLogger

def arg_reg_problem(prefix=''):
    """
    Command line parser for regression problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("gnn")
    gp.add("-epochs", type=int, default=10,
           help='Number of training epochs')
    gp.add("-batch_size", type=int, default=4,
           help="Size of batches.")
    gp.add("-particles", type=int, default=100,
           help='Number of simulation particles')
    gp.add("-samples", type=int, default=5,
           help='Number of simulations per dataset')
    gp.add("-sim_len", type=int, default=1000,
           help='Length of simulations')
    gp.add("-dim", type=int, default=2,
           help='Dimensionality of simulations')
    gp.add("-radius", type=float, default=0.1,
           help='radius to connect points in the graph')
    gp.add("-seq_len", type=int, default=5,
           help="Sequence length of input to timestepper")
    gp.add("-horizon", type=int, default = 10,
           help="Number of training prediction steps")
    gp.add("-latent_size", type=int, default=16,
           help="Size of latent dimesnion")
    gp.add("-hidden_size", type=int, default = 32,
           help="Size of hidden layers")
    gp.add("-mp_layers", type=int, default=2,
           help="Number of message passing layers")
    gp.add("-integrator", type=str, default="Euler",
           help="Integration scheme")
    
    return parser

def gen_data(points=100, sim_len=1000, samples=1):
    initial_coo = torch.rand((samples,points,2))
    vel = 0.01 * torch.rand((samples,points,2))
    
    positions = [initial_coo]
    for l in range(sim_len):
        pos = positions[-1]
        dist = torch.cdist(pos[-1],pos[-1]).fill_diagonal_(1.0)
        idist = 1.0/dist
        idist.clip_(0.0,1000)
        vect = pos.view(samples, points, 1, 2) - pos.view(samples, 1, points, 2)
        force = ((0.000001*idist) @ vect).sum(-2)
        #torch.clip_(force, min = -, max=+)
        vel = 0.90*vel + force
        new_pos = pos + vel
        
        #Rebound off left/bottom:
        low_bounds = torch.argwhere(new_pos < 0)
        new_pos[low_bounds[:,0],low_bounds[:,1],low_bounds[:,2]] = 0 - new_pos[low_bounds[:,0],low_bounds[:,1],low_bounds[:,2]]
        vel[low_bounds[:,0],low_bounds[:,1],low_bounds[:,2]] = -vel[low_bounds[:,0],low_bounds[:,1],low_bounds[:,2]]
        #Reobund off right/top
        up_bounds = torch.argwhere(new_pos > 1)
        new_pos[up_bounds[:,0],up_bounds[:,1],up_bounds[:,2]] = new_pos[up_bounds[:,0],up_bounds[:,1],up_bounds[:,2]] - 1
        vel[up_bounds[:,0],up_bounds[:,1],up_bounds[:,2]] = -vel[up_bounds[:,0],up_bounds[:,1],up_bounds[:,2]]
    
        positions.append(new_pos)   
    positions = torch.stack(positions, -2)
    return positions

def get_dataloaders(train, dev, test, radius=0.1, seq_len=5, horizon=10, batch_size=1):
    train_dset = GraphDataset(node_attr={'position': list(train)},
                              seq_len = seq_len,
                              seq_horizon = horizon,
                              seq_stride = 1,
                              build_graphs = "position",
                              connectivity_radius = radius,
                              graph_self_loops = True,
                              name="train")
    train_dset = DataLoader(train_dset,
                            batch_size = batch_size,
                            shuffle = True,
                            collate_fn = GraphDataset.collate_fn)
    dev_dset = GraphDataset(node_attr={'position': list(dev)},
                              seq_len = seq_len,
                              seq_horizon = horizon,
                              seq_stride = 1,
                              build_graphs = "position",
                              connectivity_radius = radius,
                              graph_self_loops = True,
                              name='dev')
    dev_dset = DataLoader(dev_dset,
                            batch_size = batch_size,
                            shuffle = True,
                            collate_fn = GraphDataset.collate_fn)
    sim_len = test[0].shape[1]
    test_dset = GraphDataset(node_attr={'position': list(test)},
                              seq_len = seq_len,
                              seq_horizon = sim_len - seq_len,
                              seq_stride = sim_len,
                              build_graphs = "position",
                              connectivity_radius = radius,
                              graph_self_loops = True,
                              name='test')
    test_dset = DataLoader(test_dset,
                             batch_size = 1,
                             shuffle = False,
                             collate_fn = GraphDataset.collate_fn)
    return train_dset, dev_dset, test_dset

def get_loss():
    Y_True = variable("y_position")
    Y_Pred = variable("Y_pred_graph_timestepper")
    mse_loss = ((Y_True == Y_Pred) ^ 2)
    mse_loss.name = 'mse_loss'
    loss = PenaltyLoss([mse_loss],[])
    return loss
    
class PreProcessor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        node_attr = data['position']
        edge_index = data['edge_index']
        
        relative_displacement = (
            node_attr[edge_index[0], -1] - node_attr[edge_index[1], -1])
        relative_distance = torch.norm(
            relative_displacement, dim=1, keepdim=True)
        edge_attr = torch.cat(
            [relative_displacement, relative_distance], axis=1)
        
        return {
            'node_attr': node_attr.reshape(node_attr.shape[0],-1),
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        }
        
if __name__ == "__main__":
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_reg_problem()])
    args, grps = parser.parse_arg_groups()
    
    """Create Datasets"""
    train_data = gen_data(points = args.particles, sim_len = args.sim_len, samples=args.samples)
    dev_data = gen_data(points = args.particles, sim_len = args.sim_len, samples = args.samples)
    test_data = gen_data(points = args.particles, sim_len = args.sim_len, samples = 1)
    train_dset, dev_dset, test_dset = get_dataloaders(train_data, dev_data, test_data, 
                                                      args.radius, args.seq_len, args.horizon, args.batch_size)

    """Build Model"""
    preprocessor = PreProcessor()
    model = GraphTimestepper(
        num_node_features = args.dim * args.seq_len,
        num_edge_features = args.dim + 1,
        out_size = args.dim,
        latent_size = args.latent_size,
        hidden_sizes = [args.hidden_size],
        message_passing_steps = args.mp_layers,
        preprocessor = preprocessor,
        integrator = args.integrator,
        input_key_map={'x0':'position', 'Yf':'y_position'}
    )
    components = [model]
    
    loss = get_loss()
    problem = Problem(components, loss)
    #problem.plot_graph()
    
    optimizer = torch.optim.AdamW(problem.parameters())
    
    """
    Metrics and Logger
    """
    args.savedir = 'test_regression'
    args.verbosity = 1
    metrics = ["train_loss", "dev_loss"]
    logger = MLFlowLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'test_regression'
    
    """
    Train and Test
    """
    trainer  = Trainer(
        problem,
        train_dset,
        dev_dset,
        test_dset,
        optimizer,
        logger=logger,
        epochs = args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )
    best_model = trainer.train()
