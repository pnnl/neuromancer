
from neuromancer import dataset

# loading arbitrary data from a CSV file
data = dataset.read_file("../../../psl/psl/datasets/EED_building/EED_building.csv")
train, val, test = dataset.split_data(data)
train, stats = dataset.normalize_data(train, "zscore")
val, _ = dataset.normalize_data(val, "zscore", stats=stats)
test, _ = dataset.normalize_data(test, "zscore", stats=stats)
train_dset = dataset.SequenceDataset(train, nsteps=16)
nstep, loop, dims = dataset.get_sequence_dataloaders(data, 16, norm_type="zscore")
train_nstep, val_nstep, test_nstep = nstep
next(iter(train_nstep))