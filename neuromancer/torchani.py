

import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import pickle
# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

import torch.nn.functional as F
import slim
from neuromancer import blocks, arg
from neuromancer.activations import activations
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.loggers import BasicLogger, MLFlowLogger



parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.loss(), arg.lin()])
grp = parser.group('OPTIMIZATION')
# TODO: include metric
grp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
        help="Metric for model selection and early stopping.")
args, grps = parser.parse_arg_groups()
print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
grp = parser.group('DATA')
grp.add("-data_path", type=str, default='../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
grp.add("-pickled_data", type=str, default='dataset.pkl')

device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"


###############################################################################

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
species_order = ['H', 'C', 'N', 'O']
num_species = len(species_order)
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)

###############################################################################

batch_size = 2560
dspath = args.data_path
pickled_dataset_path = args.pickled_data

# We pickle the dataset after loading to ensure we use the same validation set
# each time we restart training, otherwise we risk mixing the validation and
# training sets on each restart.
if os.path.isfile(pickled_dataset_path):
    print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
    with open(pickled_dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    training = dataset['training'].collate(batch_size).cache()
    validation = dataset['validation'].collate(batch_size).cache()
    energy_shifter.self_energies = dataset['self_energies'].to(device)
else:
    print(f'Processing dataset in {dspath}')
    training, validation = torchani.data.load(dspath)\
                                        .subtract_self_energies(energy_shifter, species_order)\
                                        .species_to_indices(species_order)\
                                        .shuffle()\
                                        .split(0.8, None)
    with open(pickled_dataset_path, 'wb') as f:
        pickle.dump({'training': training,
                     'validation': validation,
                     'self_energies': energy_shifter.self_energies.cpu()}, f)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# When iterating the dataset, we will get a dict of name->property mapping
#
###############################################################################
# Now let's define atomic neural networks.
aev_dim = aev_computer.aev_length

def h_network(aev_dim,
                outsize=1,
                bias=True,
                linear_map=slim.Linear,
                nonlin=torch.nn.CELU,
                hsizes=[160, 128, 96],
                linargs=dict()):
    return blocks.MLP(aev_dim,
        outsize,
        bias=bias,
        linear_map=linear_map,
        nonlin=nonlin,
        hsizes=hsizes,
        linargs=linargs)

def c_network(aev_dim,
                outsize=1,
                bias=True,
                linear_map=slim.Linear,
                nonlin=torch.nn.CELU,
                hsizes=[144, 112, 96],
                linargs=dict()):
    return blocks.MLP(aev_dim,
        outsize,
        bias=bias,
        linear_map=linear_map,
        nonlin=nonlin,
        hsizes=hsizes,
        linargs=linargs)

def n_network(aev_dim,
                outsize=1,
                bias=True,
                linear_map=slim.Linear,
                nonlin=torch.nn.CELU,
                hsizes=[128, 112, 96],
                linargs=dict()):
    return blocks.MLP(aev_dim,
        outsize,
        bias=bias,
        linear_map=linear_map,
        nonlin=nonlin,
        hsizes=hsizes,
        linargs=linargs)

def o_network(aev_dim,
                outsize=1,
                bias=True,
                linear_map=slim.Linear,
                nonlin=torch.nn.CELU,
                hsizes=[128, 112, 96],
                linargs=dict()):
    return blocks.MLP(aev_dim,
        outsize,
        bias=bias,
        linear_map=linear_map,
        nonlin=nonlin,
        hsizes=hsizes,
        linargs=linargs)


nn = torchani.ANIModel([h_network(aev_dim=aev_dim), c_network(aev_dim=aev_dim),
                        n_network(aev_dim=aev_dim), o_network(aev_dim=aev_dim)])
print(nn)


###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torchani.nn.Sequential(aev_computer, nn).to(device)

"""
# TODO: write a wrapper for nn.Module that turs it into component
# include input and output keys!
"""

class Component(torch.nn.Module):
    def __init__(self, module, input_keys, output_keys):
        super().__init__()
        self.module = module
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, data):
        output = self.module(*[data[k] for k in self.input_keys])
        if not type(output) is tuple:
            output = (output, )
        output = {k: v for k,v in zip(self.output_keys, output)}
        return output


###############################################################################

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

###############################################################################

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))


###############################################################################
# Finally, we come to the training loop.

"""
# TODO: define neuromancer problem with torchani component
# TODO: define callback for evaluation 
"""

mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 10
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
