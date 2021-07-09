from collections import namedtuple
import collections

import torch
import torchani
import os
import pickle

import torch.nn.functional as F
import slim
from neuromancer import blocks, arg
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.loggers import MLFlowLogger
from neuromancer.datasets import DataDict


class Component(torch.nn.Module):
    """
    Note that this is a little different from the normal component interface. Here we use input keys to get
    values from the data dictionary and package them as a tuple to hand to the nn.Module (required by the torchani
    module). Normally we would unpack the tuple with * when handed to the nn.Module for arbitrary number of args.
    The standard Component wrapper will therefore have one additional character in the call to self.module, i.e.

    output = self.module(*[data[k] for k in self.input_keys])
    """
    def __init__(self, module, input_keys, output_keys, device):
        """
        :param module: (torch.nn.Module)
        :param input_keys:
        :param output_keys:
        :param device:
        """
        super().__init__()
        self.module = module
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.device = device

    def forward(self, data):
        """

        :param data:
        :return:
        """
        output = self.module([data[k].to(self.device) for k in self.input_keys])
        if not isinstance(output, collections.abc.Sequence):
            output = (output, )
        output = {k: v for k, v in zip(self.output_keys, output)}
        return output


class WrapAni:
    """This is a wrapper since our Problem expects all data dictionaries handed to it as input to have a name.
       Takes an iterable object (assumed to hand back dictionaries during iteration). Overwrites iterable behavior to
       give the list of dictionaries being iterated over some name chosen by the user. """
    def __init__(self, iterable, name):
        self.iterable = iterable
        self.name = name

    def __iter__(self):
        for k in self.iterable:
            data = DataDict(k)
            data.name = self.name
            yield data

    def __len__(self):
        return len(self.iterable)


def get_ani_data(species_order, device, batch_size, data_path, pickled_data):
    """

    :param species_order: (list of str) List of characters indicating atom types with position indicating integer index
    :param device: (str) String indicating device to place computation and data on
    :param batch_size: (int) Mini-batch size.
    :param data_path: (str) Path to ANI dataset file
    :param pickled_data: (str) Path to previously processed pickled ANI data.

    :return:ANIDataset object which is a wrapper conforming data to what neuromancer expects.

    We pickle the dataset after loading to ensure we use the same validation set
    each time we restart training, otherwise we risk mixing the validation and
    training sets on each restart.

    When iterating the dataset, we will get a dict of name->property mapping

    """

    energy_shifter = torchani.utils.EnergyShifter(None)

    if os.path.isfile(pickled_data):
        print(f'Unpickling preprocessed dataset found in {pickled_data}')
        with open(pickled_data, 'rb') as f:
            dataset = pickle.load(f)
        training = dataset['training'].collate(batch_size).cache()
        validation = dataset['validation'].collate(batch_size).cache()
        test = dataset['test'].collate(batch_size).cache()
        energy_shifter.self_energies = dataset['self_energies'].to(device)
    else:
        print(f'Processing dataset in {data_path}')
        training, validation, test = torchani.data.load(data_path)\
                                            .subtract_self_energies(energy_shifter, species_order)\
                                            .species_to_indices(species_order)\
                                            .shuffle()\
                                            .split(0.8, 0.1, None)
        with open(pickled_data, 'wb') as f:
            pickle.dump({'training': training,
                         'validation': validation,
                         'test': test,
                         'self_energies': energy_shifter.self_energies.cpu()}, f)
        training = training.collate(batch_size).cache()
        validation = validation.collate(batch_size).cache()
        test = test.collate(batch_size).cache()

    training.name = 'train'
    validation.name = 'dev'
    test.name = 'test'
    AniDataset = namedtuple('AniDataset', ['train_data', 'dev_data', 'test_data'])
    dataset = AniDataset(WrapAni(training, 'train'),
                         WrapAni(validation, 'dev'),
                         WrapAni(test, 'test'))

    print('Self atomic energies: ', energy_shifter.self_energies)
    return dataset


def get_aev_computer(device, species_order):
    """

    :param device: (str) String indicating device to place computation on.
    :param species_order: (tuple of str) Tuple of strings with positions indicating integer index of atom type.
    :return: (nn.Module) torchani.AEVComputer which translates atom coordinates (shape=[batchsize, numatoms, 3])
    into atomic environment vectors (shape=[batchsize, numatoms, AEV-size]) for each atom in a molecule
    """
    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=device)
    ShfR = torch.tensor(
        [9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00,
         2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00,
         4.6625000e+00, 4.9312500e+00], device=device)
    Zeta = torch.tensor([3.2000000e+01], device=device)
    ShfZ = torch.tensor(
        [1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00,
         2.9452431e+00], device=device)
    EtaA = torch.tensor([8.0000000e+00], device=device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
    num_species = len(species_order)
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    return aev_computer


def get_atomic_neural_network(aev_computer):
    """

    :param aev_computer:
    :return: (torch.nn.Module) torchani.ANIModel which takes AEVs (size=(batchsize, num_atoms, AEV-size)) and outputs
                               Potential energy for the molecules (size=(batchsize)).

    """
    aev_dim = aev_computer.aev_length
    h_network = blocks.MLP(aev_dim, outsize=1, bias=True, linear_map=slim.Linear,
                           nonlin=torch.nn.CELU, hsizes=[160, 128, 96], linargs=dict())

    c_network = blocks.MLP(aev_dim, 1, bias=True, linear_map=slim.Linear,
                           nonlin=torch.nn.CELU, hsizes=[144, 112, 96], linargs=dict())

    n_network = blocks.MLP(aev_dim, outsize=1, bias=True, linear_map=slim.Linear,
                           nonlin=torch.nn.CELU, hsizes=[128, 112, 96], linargs=dict())

    o_network = blocks.MLP(aev_dim, outsize=1, bias=True, linear_map=slim.Linear,
                           nonlin=torch.nn.CELU, hsizes=[128, 112, 96], linargs=dict())

    nn = torchani.ANIModel([h_network, c_network, n_network, o_network])

    return nn


def get_args():
    """

    :return: (argparse.Namespace, str) A tuple containing the parsed command line arguments and a string indicating
                                       the device to place computations on.
    """

    parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.loss(), arg.lin()])
    grp = parser.group('DATA')
    grp.add("-data_path", type=str, default='../../../../torchani/dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
    grp.add("-pickled_data", type=str, default='dataset.pkl')
    grp.add("-batch_size", type=int, default=2560)
    args, grps = parser.parse_arg_groups()
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    return args, device


if __name__ == '__main__':
    args, device = get_args()
    species_order = ['H', 'C', 'N', 'O']

    aev_computer = get_aev_computer(device, species_order)

    nn = get_atomic_neural_network(aev_computer)

    model = torchani.nn.Sequential(aev_computer, nn).to(device)

    def energy_mse(true, pred, species):
        """
        Objective function from torchani training code.

        :param true: (torch.Tensor, shape=[batchsize], dtype=)
        :param pred: (torch.Tensor, shape=[batchsize], dtype=)
        :param species: (torch.Tensor, shape=[batchsize, num_atoms])
        :return:
        """
        num_atoms = (species.to(device) >= 0).sum(dim=1, dtype=pred.dtype)
        energy_mse = (F.mse_loss(true.to(device).type(torch.float32), pred)/torch.sqrt(num_atoms)).mean()
        return energy_mse

    hartree2kcalmol = Objective(["energies", "predicted_energies", "species"],
                                 energy_mse,
                                 weight=1.0, name="energy_mse")

    model = Problem([hartree2kcalmol], [],
                    [Component(model,
                               ['species', 'coordinates'],
                               ['out_species', 'predicted_energies'],
                               args.gpu)])
    logger = MLFlowLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                          stdout=['mean_train_energy_mse', 'mean_dev_energy_mse'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = get_ani_data(species_order, device, args.batch_size, args.data_path, args.pickled_data)

    trainer = Trainer(model, dataset, optimizer, epochs=args.epochs, logger=logger,
                      train_metric='train_energy_mse',
                      dev_metric='dev_energy_mse', test_metric='test_energy_mse',
                      eval_metric='mean_dev_energy_mse', patience=args.patience, warmup=args.warmup, lr_scheduler=True)

    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    print({k: v.shape for k, v in best_outputs.items()})






