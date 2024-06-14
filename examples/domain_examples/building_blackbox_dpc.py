import neuromancer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from neuromancer.psl.building_envelope import BuildingEnvelope
import functools
from neuromancer.dataset import DictDataset
from tqdm import tqdm
import random
from torch import tensor
import torch
from neuromancer.psl.signals import signals
import numpy as np
from torch.utils.data import DataLoader
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dynamics.integrators import RK4
from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import yaml
from dpc_util import simulate, stack_refs, load_stats, remove_key_prefix

# remove problematic signal
if 'beta' in signals:
  del signals['beta']

# set the seed
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GradBuildingEnvelope(BuildingEnvelope):
    def __init__(self, device=None, *args, **kwargs):
        self.device=device
        super().__init__(*args, **kwargs)

    def get_q(self, u):
        m_flow = u[0:self.n_mf]
        dT = u[self.n_mf:self.n_mf + self.n_dT]
        q = m_flow * self.rho * self.cp * self.time_reg * dT
        return q

systems = {system: functools.partial(GradBuildingEnvelope, system=system) for system in BuildingEnvelope.systems}

"""Choose a system."""

# supported systems: ['SimpleSingleZone', 'Reno_full']
psl_key = 'SimpleSingleZone'
psl_sys = systems[psl_key]()

# select appropriate device for runtime
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ny, nu, nd = psl_sys.ny, psl_sys.nU, psl_sys.nD_obs
hsize = 128
nlayers = 2
ssm = blocks.MLP(
    ny + nu + nd,
    ny,
    bias=True,
    linear_map=torch.nn.Linear,
    nonlin=activations['relu'],
    hsizes=[hsize for _ in range(nlayers)],
)
ssm = RK4(ssm, h=torch.tensor(0.01))
system_node = Node(ssm, ["yn", "U", "D"], ["yn"])
system = System([system_node], name='NSSM').to(device)
system.nstep_key = 'Y'

xpred = variable('yn')[:, :-1, :]
xtrue = variable('Y')

loss = (xpred == xtrue) ^ 2
loss.update_name('mse')

obj = PenaltyLoss([loss], [])
problem = Problem([system], obj).to(device)
problem.show()

class StridedDataset(Dataset):
    """
    Strided Sequence Dataset compatible with neuromancer Trainer

    This dataset generates subsequences of a fixed length from a sequence dataset.
    The goal is to decouple the prediction horizon length from the length of the rollout.
    This is useful as a form of data augmentation.

    Contributors:
        - @Seth1Briney
        - @HarryLTS
        - @diego-llanes
    """

    def __init__(self, datadict, L=32, name='train', stride=1, update_fn=None):
        """
        :rtype: object
        :param datadict: (dict {str: Tensor}) Dictionary of tensors with shape (N, T, D)
        :param name: (str) Name of dataset
        :param L (int) Length of each subsequence
        :param stride (int) Stride between subsequences
        :param update_fn (callable) Function to collect the first element of the sequence for a rollout of predictions
        :example of update_fn: lambda d: d["yn"] = d["Y"][0:1, :]
        """
        super().__init__()
        self.datadict = datadict
        self.L = L
        self.name = name
        self.nsim, self.nsteps, _ = next(iter(datadict.values())).shape
        self.seqs_per_sim = self.nsteps - self.L + 1
        self.length = (self.nsim * self.seqs_per_sim) // stride
        self.update_fn = update_fn
        self.stride = stride

    def __getitem__(self, i):
        """Fetch a single item from the dataset."""
        i_sim, i = self.remainder(i * self.stride, self.seqs_per_sim)
        sim_chunk = {k: v[i_sim][i: i+self.L] for k, v in self.datadict.items()}
        if self.update_fn:
            self.update_fn(sim_chunk)
        return sim_chunk

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        """Wraps the default PyTorch batch collation function and adds a name field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        batch['name'] = self.name
        return batch

    def remainder(self, n, d):
        """ use the remainder algorithm to index sequences
        :param i (int) index of subsequence
        """
        i_sim = n // d
        i = n % d
        return i_sim, i

NV = load_stats(psl_sys, psl_key)
NVD = {k: v.to(device) for k, v in NV.items()}

def denorm(x, key, cpu=True):
  normer = NV if cpu else NVD
  return normer[f'{key}_min'] + (normer[f'{key}_max'] - normer[f'{key}_min']) * (x+1)/2

def norm(x, key, cpu=True):
  normer = NV if cpu else NVD
  return 2 * (x - normer[f'{key}_min']) / (normer[f'{key}_max'] - normer[f'{key}_min']) - 1

def get_dataset(name, nsim=20, nsteps=3500, L=512, stride=509, device='cpu'):
    data = simulate(psl_sys, NV, nsim=nsim, nsteps=nsteps)
    def update_yn(d):
        d['yn'] = d['Y'][0:1, :].clone().detach().requires_grad_(True)

    if name == 'train':
        return StridedDataset(
            {
                'Y': norm(data['Y'], 'Y').to(device),
                'U': norm(data['U'], 'U').to(device),
                'D': norm(data['D'], 'D').to(device)
            },
            name=name,
            update_fn=update_yn,
            L=L,
            stride=stride
        )
    else:
        return DictDataset(
            {
                'yn': norm(data['Y'][:, 0:1, :], 'Y'),
                'Y': norm(data['Y'], 'Y').to(device),
                'U': norm(data['U'], 'U').to(device),
                'D': norm(data['D'], 'D').to(device)
            },
            name=name
        )

train_data = get_dataset('train', nsim=100, device=device)
dev_data = get_dataset('dev', nsim=20, device=device)

num_colors = 6
pred_colors = [plt.cm.winter(i / num_colors) for i in range(num_colors)]
target_colors = [plt.cm.ocean(i / num_colors) for i in range(num_colors)]
control_colors = [plt.cm.plasma(i / num_colors) for i in range(num_colors)]


sample_control = denorm(train_data.datadict['U'][0].cpu(), key='U').numpy()
sample_sim = denorm(train_data.datadict['Y'][0].cpu(), key='Y').numpy()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for v in range(sample_sim.shape[1]):
  axs[0].plot(sample_sim[:, v], label=f'Area {v+1} Temperature', color=target_colors[v])
axs[0].set_title('Building Temperature')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Temperature (°C)')
axs[0].legend()

lns = []
for v in range(sample_control.shape[1] - 1):
    ln = axs[1].plot(sample_control[:, v], label=f'Mass Flow {v+1}', color=control_colors[v])[0]
    lns.append(ln)

axs[1].set_title('Control')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Mass Flow (kg/s)')


ax2 = axs[1].twinx()
lns.append(ax2.plot(sample_control[:, -1], label='Δ Temperature', color='crimson')[0])
ax2.set_ylabel('Δ Temperature (°C)')

labs = [ln.get_label() for ln in lns]
ax2.legend(lns, labs, loc=1)

plt.tight_layout()
plt.show()

train_loader = DataLoader(
    train_data,
    batch_size=128,
    collate_fn=train_data.collate_fn,
    shuffle=True
)

dev_loader = DataLoader(
    dev_data,
    batch_size=128,
    collate_fn=dev_data.collate_fn,
    shuffle=False
)

logger = BasicLogger(savedir='ssm_out', stdout=['train_loss', 'dev_loss'], verbosity=1)

# initialize optimizer and trainer
opt = optim.Adam(system.parameters(), 1e-3)
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=opt,
    epochs=20,
    patience=10,
    logger=logger,
    train_metric="train_loss",
    eval_metric="dev_loss",
    lr_scheduler=True,
    device=device
)

# train the model
best_model = trainer.train()
trainer.model.load_state_dict(best_model)
problem.to("cpu")

# evaluate the system
test_data_dict = get_dataset('test', nsim=5).datadict
system_node.eval()
test_out = system(test_data_dict)

# discard last predicted state
test_out['yn'] = test_out['yn'][:, :-1, :]

import matplotlib.pyplot as plt

def plot_sysid(data, sim=0):
    plt.clf()
    plot = {}
    for key in data:
        norm_key = 'Y' if key == 'yn' else key
        plot[key] = denorm(data[key], norm_key).detach().cpu().numpy()

    sample_pred = plot["yn"][sim, :, :]
    sample_target = plot["Y"][sim, :, :]
    sample_control = plot["U"][sim, :, :]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    for v in range(sample_pred.shape[1]):
      axs[0].plot(sample_pred[:, v], label=f'Area {v+1} Prediction', color=pred_colors[v])
      axs[0].plot(sample_target[:, v], label=f'Area {v+1} Reference', color=target_colors[v], linestyle='--')

    axs[0].set_title("SSM vs. Real System")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].legend()

    lns = []
    for v in range(sample_control.shape[1] - 1):
        ln = axs[1].plot(sample_control[:, v], label=f'Mass Flow {v+1}', color=control_colors[v])[0]
        lns.append(ln)

    axs[1].set_title('Control')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mass Flow (kg/s)')


    ax2 = axs[1].twinx()
    lns.append(ax2.plot(sample_control[:, -1], label='Δ Temperature', color='crimson')[0])
    ax2.set_ylabel('Δ Temperature (°C)')

    labs = [ln.get_label() for ln in lns]
    ax2.legend(lns, labs, loc=1)

    axs[1].set_title('Control')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Control Values')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


plot_sysid(test_out, sim=2)

def sample_min_max(
        nsim=20,
        prob=0.0005,
        bounds=[(20, 22), (18, 24.4)],
        nsteps=3500
        ):
    """
    :param growth_rate: (float) The rate at which the cumulative probability grows at each timestep. Default is 0.00001.
    :param max_prob: (float) The maximum probability value for changing the bounds. Default is 0.5.
    :param bounds: (list[tuple[float, float]]) A list of tuples where each tuple has two float values that represent the bounds. Default is [(20, 22), (18, 24.4)].

    :return: (numpy.ndarray) A 3D array with shape (nsim, nsteps, 2). The last dimension contains the min and max bounds respectively for each timestep in the simulation.
    """
    y_min_list = []
    y_max_list = []

    for _ in range(nsim):
        current_bound_index = random.randint(0, 1)
        y_min = np.full((nsteps,), bounds[current_bound_index][0], dtype=np.single)
        y_max = np.full((nsteps,), bounds[current_bound_index][1], dtype=np.single)

        for t in range(nsteps):
            if random.random() < prob:
                current_bound_index = 1 - current_bound_index
                y_min[t:] = bounds[current_bound_index][0]
                y_max[t:] = bounds[current_bound_index][1]

        y_min_list.append(y_min)
        y_max_list.append(y_max)
    y_min_3d = np.expand_dims(np.stack(y_min_list), axis=2)
    y_max_3d = np.expand_dims(np.stack(y_max_list), axis=2)
    return np.concatenate([y_min_3d,y_max_3d], axis=2)

"""We define functions to generate the conditions necessary for running and evaluating control. The bound lookahead $p$ specifies the number of timesteps into the future that our policy takes into account for the bounds."""

bound_lookahead = 3

def get_control_dataset(name, nsim=20, nsteps=3500, device='cpu'):
    data = simulate(psl_sys, NV, nsim=nsim, nsteps=nsteps)
    refs = sample_min_max(nsim=nsim, nsteps=nsteps)

    data['Y'] = data['Y'][:, 0:1, :]
    data['LB'] = refs[:, :, 0:1]
    data['UB'] = refs[:, :, 1:2]

    bs = data['LB'].shape[0]
    ep_len = data['LB'].shape[1]

    return DictDataset({
            'yn': norm(data['Y'][:, 0:1, :], key='Y').to(device),
            'D': norm(data['D'], key='D').to(device),
            'LB': norm(stack_refs(data['LB'], forecast=bound_lookahead),
                                key='Y').to(device),
            'UB': norm(stack_refs(data['UB'], forecast=bound_lookahead),
                                key='Y').to(device),
            'LB_dn': stack_refs(data['LB'], forecast=bound_lookahead).to(device),
            'UB_dn': stack_refs(data['UB'], forecast=bound_lookahead).to(device),
            'xn_dn': data['X'][:, 0:1, :].to(device),
            'Dhidden_dn': data['Dhidden'].to(device)
        },
        name=name
    )

"""Now, we can generate the datasets."""

train_data = get_control_dataset('train', nsim=64, device=device)
dev_data = get_control_dataset('dev', nsim=16, device=device)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    collate_fn=train_data.collate_fn,
    shuffle=True
)

dev_loader = DataLoader(
    dev_data,
    batch_size=64,
    collate_fn=dev_data.collate_fn,
    shuffle=False
)

hsize = 128
nlayers = 2

policy = blocks.MLP_bounds(
    insize=ny + bound_lookahead * 2 + nd,
    outsize=nu,
    hsizes=[hsize for _ in range(nlayers)],
    nonlin=activations['relu'],
    linear_map=torch.nn.Linear,
    min=-1.,
    max=0.
).to(device)

losses = []

q = variable('Q')

# loss defined as difference from zero, L1 loss
energy_loss = (q == torch.tensor(0.).to(device))
energy_loss.update_name('energy_loss')
losses.append(energy_loss)

q_min = torch.sum(psl_sys.get_q(NV['U_min']))
q_max = torch.sum(psl_sys.get_q(NV['U_max']))

def u2q(U):
    norm_u = denorm(U, key='U', cpu=False).permute(1, 0)
    denorm_q = torch.sum(psl_sys.get_q(norm_u), dim=0, keepdim=True)
    norm_q = (denorm_q - q_min) / (q_max - q_min)
    return norm_q.permute(1, 0)

q_node = Node(
    u2q,
    ['U'], ['Q'],
    name='U2Q'
)

u = variable('U')

# loss defined as difference between adjacent controls
stability_loss = ((u[:,:-1,:] - u[:,1:,:]) == torch.tensor(0.).to(device))
stability_loss.update_name('stability_loss')
losses.append(stability_loss)

pred = variable('yn_dn')

ub = variable('UB_dn')
lb = variable('LB_dn')

# loss defined as aggregate of upper bound violation and lower bound violation
# losses are multiplied by the number of areas to sum their violations
state_upper_bound_penalty = float(ny) * (pred < ub[:, :, 0:1])
state_lower_bound_penalty = float(ny) * (pred > lb[:, :, 0:1])

constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty
]

denormnode = Node(
    lambda u, yn: (
    denorm(u, key='U', cpu=False), denorm(yn, key='Y', cpu=False)),
    ['U', 'yn'], ['U_dn', 'yn_dn'],
    name='denorm'
)

policy_node = Node(policy, ['yn', 'D', 'UB', 'LB'], ['U'], name='policy')

system_node.eval()
control_system = System([policy_node, system_node, q_node, denormnode], name='control_system')
control_system.nstep_key = 'UB'
control_system.to(device)

obj = PenaltyLoss(losses, constraints)
problem = Problem([control_system], obj)

logger = BasicLogger(savedir='policy_out', stdout=['train_loss', 'dev_loss'], verbosity=1)
opt = optim.Adam(policy.parameters(), 1e-3)

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=opt,
    epochs=10,
    patience=5,
    train_metric="train_loss",
    eval_metric="dev_loss",
    lr_scheduler=True,
    device=device,
    logger=logger
)

best_model = trainer.train()
trainer.model.load_state_dict(best_model)
problem.to('cpu')

def test_u2q(U):
    norm_u = denorm(U, key='U').permute(1, 0)
    denorm_q = torch.sum(psl_sys.get_q(norm_u), dim=0, keepdim=True)
    norm_q = (denorm_q - q_min) / (q_max - q_min)
    return norm_q.permute(1, 0)

test_q_node = Node(
    test_u2q,
    ['U'], ['Q'],
    name='U2Q'
)

test_denormnode = Node(
    lambda u:
    denorm(u, key='U'),
    ['U'], ['U_dn'],
    name='denorm'
)

test_normnode = Node(
    lambda yn: norm(yn, key='Y'),
    ['yn_dn'], ['yn'],
    name='norm'
)

use_pretrained = True

if use_pretrained:
    file_root = f'models/blackbox/{psl_key}'
    with open(os.path.join(file_root, 'config.yaml'), 'r') as yaml_file:
      cfg = yaml.unsafe_load(yaml_file)
    hsize = cfg.policy_hsize
    nlayers = cfg.policy_nlayers
    act = cfg.policy_act

    pt_policy = blocks.MLP_bounds(
        insize=ny + bound_lookahead * 2 + nd,
        outsize=nu,
        hsizes=[hsize for _ in range(nlayers)],
        nonlin=activations[act],
        linear_map=torch.nn.Linear,
        min=-1.,
        max=0.
    )

    problem_state_dict = torch.load(os.path.join(file_root, 'best_model_state_dict.pth'), map_location=torch.device('cpu'))

    pt_policy_node = Node(pt_policy, ['yn', 'D', 'UB', 'LB'], ['U'], name='policy')
    pt_policy_node.load_state_dict(remove_key_prefix(problem_state_dict, 'nodes.0.nodes.0.'))
    test_policy_node = pt_policy_node
else:
    test_policy_node = policy_node

psl_sys_t = systems[psl_key](backend='torch')
test_system_node = Node(psl_sys_t, ['xn_dn', 'U_dn', 'Dhidden_dn'], ['xn_dn', 'yn_dn'], name='test_system')
test_system = System([test_policy_node, test_denormnode, test_system_node, test_normnode, test_q_node], name='cl_system')
test_system.nstep_key = 'UB'

test_data_dict = get_control_dataset('test', nsim=5, device='cpu').datadict
test_system.eval()
test_system = test_system.to('cpu')
test_out = test_system(test_data_dict)

def plot_control(data, sim=0):
    plt.clf()
    plot = {}
    for key in data:
        plot[key] = data[key].detach().cpu().numpy()

    sample_temp = plot["yn_dn"][sim, :, :]
    sample_LB = plot["LB_dn"][sim, :, 0:1]
    sample_UB = plot["UB_dn"][sim, :, 0:1]
    sample_control = plot["U_dn"][sim, :, :]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    for v in range(sample_temp.shape[1]):
      axs[0].plot(sample_temp[:, v], label=f'Area {v+1} Temperature', color=target_colors[v])

    axs[0].plot(sample_LB, linestyle="--", label="Minimum Bound", color="red")
    axs[0].plot(sample_UB, linestyle="--", label="Maximum Bound", color="red")

    axs[0].set_title("Temperature vs. Constaints")
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].set_xlabel("Time")
    axs[0].legend()

    lns = []
    for v in range(sample_control.shape[1] - 1):
        ln = axs[1].plot(sample_control[:, v], label=f'Mass Flow {v+1}', color=control_colors[v])[0]
        lns.append(ln)

    axs[1].set_title('Control')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mass Flow (kg/s)')


    ax2 = axs[1].twinx()
    lns.append(ax2.plot(sample_control[:, -1], label='Δ Temperature', color='crimson')[0])
    ax2.set_ylabel('Δ Temperature (°C)')

    labs = [ln.get_label() for ln in lns]
    ax2.legend(lns, labs, loc=1)

    axs[1].set_title('Control')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Control Values')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


plot_control(test_out, sim=3)