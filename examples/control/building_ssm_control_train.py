"""
Test and validation sets are step function references
Length of test and validation sets: 1000

What to do if not learning well:
+ Increase network capacity
+ Fiddle with optimization hyperparameters
+ Fiddle with nonlinearity
+ Fiddle with normalization
+ Forecast disturbances
+ Moving horizon for y
"""
import matplotlib.pyplot as plt
import os, argparse, dill, sklearn, torch
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from neuromancer.modules.activations import activations
from neuromancer.modules import blocks
from neuromancer.psl.building_envelope import systems
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset

import numpy as np
from neuromancer.system import Node, System


def get_test_data(sys, nsim):
    """
    Gets a reference trajectory by simulating the system with random initial conditions.

    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    ny, nu, nd, ndh = sys.ny, sys.nU, 1, sys.nD
    sim = sys.simulate(nsim=nsim, x0=sys.get_x0())
    R = sys.get_R(nsim)
    # R = sim['Y']
    Dhidden = sys.get_D(nsim)
    R, Dhidden = (R.reshape(1, nsim, ny),
                  Dhidden.reshape(1, nsim, ndh))

    R = sys.normalize(R, key='Y')
    D = sys.normalize(Dhidden, key='D')[:, :, sys.d_idx]

    R, D, Dhidden = (torch.tensor(R, dtype=torch.float32),
                     torch.tensor(D, dtype=torch.float32),
                     torch.tensor(Dhidden, dtype=torch.float32))

    U_upper = torch.tensor((sys.umax - sys.stats['U']['mean'][0])/sys.stats['U']['std'][0], dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)
    U_lower = torch.tensor((sys.umin - sys.stats['U']['mean'][0])/sys.stats['U']['std'][0], dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)

    data = {'R': R, 'D': D, 'U_upper': U_upper, 'Dhidden': Dhidden, 'U_lower': U_lower, 'name': 'test'}
    # print([torch.isnan(v) for v in data.values() if isinstance(v, torch.Tensor)])
    return data


def get_data(nsteps, sys, nsim, bs, name='train'):
    """
    Gets a reference trajectory by simulating the system with random initial conditions.

    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)

    """
    R, D, Dhidden = [], [], []
    ny, nu, nd, ndh = sys.ny, sys.nU, 1, sys.nD
    for _ in range(nsim//nsteps):
        sim = sys.simulate(nsim=nsteps, x0=sys.get_x0())
        R.append(sim['Y'])
        D.append(sim['D'])
        Dhidden.append(sim['Dhidden'])
    R, D, Dhidden = sys.B.core.stack(R), sys.B.core.stack(D), sys.B.core.stack(Dhidden)
    sim = sys.simulate(nsim=(nsim//nsteps) * nsteps, x0=sys.get_x0())

    r, d, dhidden = (sim['Y'].reshape(nsim//nsteps, nsteps, ny),
                     sim['D'].reshape(nsim//nsteps, nsteps, nd),
                     sim['Dhidden'].reshape(nsim//nsteps, nsteps, ndh))
    R, D, Dhidden = (np.concatenate([R, r], axis=0),
                     np.concatenate([D, d], axis=0),
                     np.concatenate([Dhidden, dhidden], axis=0))
    R = sys.normalize(R, key='Y')
    Dhidden = sys.normalize(Dhidden, key='D')
    D = Dhidden[:, :, sys.d_idx]

    R, D, Dhidden = (torch.tensor(R, dtype=torch.float32),
                     torch.tensor(D, dtype=torch.float32),
                     torch.tensor(Dhidden, dtype=torch.float32))
    R += 0.01*torch.randn(R.shape)

    U_upper = torch.tensor((sys.umax - sys.stats['U']['mean'][0])/sys.stats['U']['std'][0], dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)
    U_lower = torch.tensor((sys.umin - sys.stats['U']['mean'][0])/sys.stats['U']['std'][0], dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)

    data = DictDataset({'R': R, 'D': D, 'Dhidden': Dhidden, 'U_upper': U_upper, 'U_lower': U_lower}, name=name)
    loader = DataLoader(data, batch_size=bs, collate_fn=data.collate_fn, shuffle=True)

    return loader


def psl_init(data):
    data['xn'] = sys.x0.reshape(1, 1, sys.nx0)
    data['yn_true'] = (sys.C @ sys.x0 + sys.F.ravel() - sys.y_ss).reshape(1, 1, sys.ny)
    # data['R'][:, :32, :] = sys.normalize(data['yn_true'], key='Y') + 0.001
    # data['R'][:, :32, :] = sys.normalize(data['yn_true'], key='Y') + 0.001
    return data


def ssm_train_init(data):
    data = {k: v for k, v in data.items()}
    data['Y'] = data['R']
    data['yn'] = data['R'][:, 0:1, :]
    data['yn'] += data['yn'] + torch.randn(data['yn'].shape)
    return data


def ssm_test_init(data):
    data = {k: v for k, v in data.items()}
    data['Y'] = data['R']
    data['yn'] = data['R'][:, 0:1, :] + 0.0001
    return data


class Policy(nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.net = blocks.MLP(insize, outsize, bias=True, linear_map=nn.Linear, nonlin=activations[args.act],
                              hsizes=[args.hsize for h in range(args.nlayers)])

    def forward(self, y, D, U_upper, U_lower, R):
        features = torch.cat([y, D, U_upper, U_lower, R], dim=-1)
        return self.net(features)
        if self.training:
            return self.net(features)
        else:
            return torch.clamp(self.net(features),
                               min=torch.tensor((0.-sys.stats['U']['mean'])/sys.stats['U']['std'].flatten(), dtype=torch.float32),
                               max=torch.tensor((5000.-sys.stats['U']['mean'])/sys.stats['U']['std'].flatten(), dtype=torch.float32))


def get_system(sys, ssm, forecast):
    """
    ny = observable state size
    ny*forecast = reference preview size
    nd = disturbance dimension
    2*nu = Upper and lower bounds on controls

    :param ny:
    :param nu:
    :param nd:
    :param forecast:
    :return:
    """
    """
    Inputs to the control policy:
    System state at current time or current history
    Disturbance at current time
    Constraints at current time
    Reference with forecast
    """
    ny, nu, nd = sys.ny, sys.nU, 1
    insize = 2*ny + nd + 2*nu
    policy = Policy(insize, nu)
    policy_node = Node(policy, ['yn', 'D', 'U_upper', 'U_lower', 'R'], ['U'])
    system_node = ssm

    system = System([policy_node, system_node], init_func=ssm_train_init, nsteps=args.nsteps)
    return system, policy_node


def get_test_system(sys, policy):
    system_node = Node(sys, ['xn', 'Udenorm', 'Dhidden'], ['xn', 'yn_true'])
    denorm_node = Node(lambda u: torch.clamp(sys.denormalize(u, key='U'), min=0., max=5000.), ['U'], ['Udenorm'])
    # denorm_node = Node(lambda u: sys.denormalize(u, key='U'), ['U'], ['Udenorm'])

    norm_node = Node(lambda y: sys.normalize(y, key='Y'), ['yn_true'], ['yn'])
    system = System([norm_node, policy, denorm_node, system_node], init_func=psl_init, nsteps=args.nsim)
    return system


def plot_results(data, prefix, figname, test=False):
    pred_traj = data[f'{prefix}R'][:, :-args.forecast, :].detach().numpy().reshape(-1, sys.ny)
    if test:
        true_traj = data[f'{prefix}yn'].detach().numpy().reshape(-1, sys.ny)
    else:
        true_traj = data[f'{prefix}yn'][:, 1:, :].detach().numpy().reshape(-1, sys.ny)
    if test:
        u = data['Udenorm'].detach().numpy().reshape(-1, sys.nU)
    else:
        u = data[f'{prefix}U'].detach().numpy().reshape(-1, sys.nU)
    hi = data[f'{prefix}U_upper'][:, :-args.forecast, :].detach().numpy().reshape(-1, sys.nU)
    lo = data[f'{prefix}U_lower'][:, :-args.forecast, :].detach().numpy().reshape(-1, sys.nU)

    np.save(os.path.join(args.logdir, f'ssm_{figname}.npy'), true_traj)
    np.save(os.path.join(args.logdir, f'ref_{figname}.npy'), pred_traj)

    pred_traj = sys.denormalize(pred_traj, key='Y')
    true_traj = sys.denormalize(true_traj, key='Y')
    if not test:
        u = sys.denormalize(u, key='U')
    hi = sys.denormalize(hi, key='U')
    lo = sys.denormalize(lo, key='U')

    mae = metrics.mean_absolute_error(true_traj, pred_traj)
    mse = metrics.mean_squared_error(true_traj, pred_traj, squared=False)
    r2 = metrics.r2_score(true_traj, pred_traj)
    print(f'mae: {mae}\tmse: {mse}\tr2: {r2}')
    logger.log_metrics({f'{prefix}mse': mse,
                        f'{prefix}mae': mae,
                        f'{prefix}r2_test': r2})

    pred_traj, true_traj, u, hi, lo = pred_traj.transpose(1, 0), true_traj.transpose(1, 0), u.transpose(1, 0), hi.transpose(1, 0), lo.transpose(1, 0)

    textstr = f'MSE: {mse: .3f}\nMAE: {mae: .3f}\nR^2: {r2: .3f}'
    figsize = 25
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.xticks(fontsize=figsize)
    fig, ax = plt.subplots(sys.ny+sys.nU, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax[row]
        if row == 0:
            axe.text(0.05, 0.95, textstr, transform=axe.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, label='Response', c='c')
        axe.plot(t2, label='Reference', c='m', linestyle='dashed')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.legend(fontsize=figsize)
    labels = [f'$u_{k}$' for k in range(len(true_traj))]
    for row, (u1, uhi, ulo, label) in enumerate(zip(u, hi, lo, labels)):
        axe = ax[row+sys.ny]
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(u1, label='Control', c='k')
        axe.plot(uhi, label='Upper bound', c='r', linestyle='dashed')
        axe.plot(ulo, label='Lower bound', c='r', linestyle='dashed')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)
    plt.tight_layout()
    plt.savefig(os.path.join(args.logdir, f'{figname}.png'))
    logger.log_artifacts({})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-system', default='SimpleSingleZone', choices=[k for k in systems],
                        help='You can use any of the systems from psl.nonautonomous with this script')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs of training.')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for gradient descent.')
    parser.add_argument('-nsteps', type=int, default=4,
                        help='Prediction horizon for optimization objective. During training will roll out for nsteps from and initial condition')
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-nsim', type=int, default=1000,
                        help="The script will generate an nsim long time series for training and testing and 10 nsim long time series for validation")
    parser.add_argument('-logdir', default='test_control',
                        help='Plots and best models will be saved here. Also will be moved to the location directory for mlflow artifact logging')
    parser.add_argument("-exp", type=str, default="test",
                        help="Will group all run under this experiment name.")
    parser.add_argument("-location", type=str, default="mlruns",
                        help="Where to write mlflow experiment tracking stuff")
    parser.add_argument("-run", type=str, default="neuromancer",
                        help="Some name to tell what the experiment run was about.")
    parser.add_argument('-hsize', type=int, default=128, help='Size of hiddens states')
    parser.add_argument('-nlayers', type=int, default=4, help='Number of hidden layers for MLP')
    parser.add_argument('-act', type=str, choices=[k for k in activations],
                        help='Activation for MLP', default='softexp')
    parser.add_argument('-iterations', type=int, default=3,
                        help='How many episodes of curriculum learning by doubling the prediction horizon and halving the learn rate each episode')
    parser.add_argument('-eval_metric', type=str, default='eval_mse')
    parser.add_argument('-forecast', type=int, default=4, help='Number of lookahead steps for reference.')
    parser.add_argument('-modeldir', default='test')
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # Data setup
    sys = systems[args.system]()
    sysid_problem = torch.load('nm_models/best_model_linear_0.6_r2.pth', pickle_module=dill)
    ssm = sysid_problem.nodes[0].nodes[0]
    simulator, policy = get_system(sys, ssm, args.forecast)
    train_data = get_data(args.nsteps+args.forecast, sys, args.nsim, args.batch_size, name='train')
    dev_data = get_data(args.nsteps+args.forecast, sys, args.nsim, args.batch_size, name='dev')
    test_data = get_test_data(sys, args.nsim+args.forecast)
    opt = optim.AdamW(policy.parameters(), args.lr)

    # Loss function and constraints definitions
    tru = variable('yn')[:, 1:, :]
    ref = variable('R')[:, :-args.forecast, :]
    u_hi = variable('U_upper')[:, :-args.forecast, :]
    u_lo = variable('U_lower')[:, :-args.forecast, :]
    u = variable('U')
    loss = (ref == tru) ^ 2
    loss.update_name('loss')
    c_upper = (u < u_hi)
    c_lower = (u > u_lo)
    obj = PenaltyLoss([loss], [c_lower, c_upper])

    # Training setup and loop
    problem = Problem([simulator], obj)
    logout = ['loss']
    logger = MLFlowLogger(args, savedir=args.logdir, stdout=['train_loss', 'dev_loss'], logout=logout)
    trainer = Trainer(problem, train_data, dev_data, test_data, opt, logger,
                      epochs=args.epochs,
                      patience=args.epochs*args.iterations,
                      train_metric='train_loss',
                      dev_metric='dev_loss',
                      test_metric='test_loss',
                      eval_metric='dev_loss')
    lr = args.lr
    for i in range(args.iterations):
        print(f'training lr={lr}')
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.
        opt.param_groups[0]['lr'] = lr

    # Model testing and evaluation
    problem.eval()
    simulator.nsteps = args.nsim
    simulator.init = ssm_test_init
    output = problem(test_data)
    plot_results(output, 'test_', 'close_loop_ssm')
    sys.change_backend('torch')
    # sys.y_ss = torch.tensor(sys.y_ss, dtype=torch.float32)  # Need to make it easier to switch a psl model from numpy to torch
    system = get_test_system(sys, policy)
    out = system(test_data)
    plot_results(out, '', 'close_loop_psl', test=True)
