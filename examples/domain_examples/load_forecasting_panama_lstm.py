#%% Load forecasting on Short-term electricity load forecasting (Panama case study) (https://data.mendeley.com/datasets/byx7sztj59/1)
# Downloads and preproceses the data. Then, trains an LSTM model from Neuromancer blocks and plots the forecast.

#%% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


#%% Set numpy and torch seeds for reproducibility, and choose device automatically
torch.manual_seed(0)
np.random.seed(0)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


#%% Download data set

# Short-term electricity load forecasting (Panama case study) from Mendeley Data 
# (DOI:10.17632/byx7sztj59.1)
url = "https://data.mendeley.com/public-files/datasets/byx7sztj59/files/94ef3285-7ce4-43e2-9fd3-9cb3f592b89c/file_downloaded"
response = requests.get(url, stream=True)

file_name = "continuous_dataset.csv"
with open(file_name, "wb") as file:
    file.write(response.content)


#%% Read csv file as Pandas dataframe and set index

df = pd.read_csv(file_name)
df = df.set_index("datetime")
df.index = pd.to_datetime(df.index)
df


#%% Drop uninformative features

# Column descriptions (from https://data.mendeley.com/datasets/byx7sztj59/1/files/6952a984-f474-437f-8640-67d773caea93)
# nat_demand: National electricity load (Target or Dependent variable) (MWh)
# T2M_toc: Temperature at 2 meters in Tocumen, Panama city (ºC)
# QV2M_toc: Relative humidity at 2 meters in Tocumen, Panama city (%)
# TQL_toc: Liquid precipitation in Tocumen, Panama city liters/m2
# W2M_toc: Wind Speed at 2 meters in Tocumen, Panama city m/s
# T2M_san: Temperature at 2 meters in Santiago city ºC
# QV2M_san: Relative humidity at 2 meters in Santiago city %
# TQL_san: Liquid precipitation in Santiago city l/m2
# W2M_san: Wind Speed at 2 meters in Santiago city m/s
# T2M_dav: Temperature at 2 meters in David city ºC
# QV2M_dav: Relative humidity at 2 meters in David city %
# TQL_dav: Liquid precipitation in David city l/m2
# W2M_dav: Wind Speed at 2 meters in David city m/s
# Holiday_ID: Unique identification number integer
# holiday: Holiday binary indicator 1 = holiday, 0 = regular day
# school: School period binary indicator 1 = school, 0 = vacations

excluded_cols = ['Holiday_ID']
df = df.drop(columns=excluded_cols)


#%%
# Split the data into training, validation and test sets

seq_length = 72 # lookback = 72 hours
train_split = 0.7
val_split = 0.15
test_split = 1 - (train_split + val_split)

train_size = int(train_split * len(df))
val_size = int(val_split * len(df))

# Train set
df_train = df[:train_size]

# Val set
val_start = train_size + seq_length
val_end = val_start + val_size
df_val = df[val_start:val_end] # add gap between train and test datasets to prevent data leakage:]

# Test set
df_test = df[val_end+seq_length:] # add gap between train and test datasets to prevent data leakage:]


#%%
# Fit scaler to train data and scale train data
scaler = MinMaxScaler(feature_range=(-1,1))
train_data = scaler.fit_transform(df_train)
train_data.min(), train_data.max()


#%% Transform val and test data using the scaler parameters for train data

val_data = scaler.transform(df_val)
test_data = scaler.transform(df_test)


#%%
# Create sequential data with a arbitrary window size

def create_timeseries(data, len_sequence):
    X, y = [], []
    for i in range(len(data) - len_sequence):
        X.append(data[i:i+len_sequence])
        y.append(data[i+len_sequence][0]) # columm 0 is the load

    X, y = np.array(X), np.array(y).reshape(-1,1)
    return X, y

X_train, y_train = create_timeseries(train_data, seq_length)
X_val, y_val = create_timeseries(val_data, seq_length)
X_test, y_test = create_timeseries(test_data, seq_length)


#%%
# Create a custom dataset class for PyTorch DataLoader

from neuromancer.dataset import default_collate


class PanamaLoadDataset(Dataset):
    def __init__(self, X, y, type="Train"):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.name = type.lower()

    def __len__(self):
        return len(self.X)

    def collate_fn(self, batch):
        """Wraps the default PyTorch batch collation function and adds a name field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        batch['name'] = self.name
        return batch

    def __getitem__(self, index):
        
        if self.name == 'test':
            return {
                'X': self.X[index],
                'Y': self.y[index],
                'name': self.name
            }
        else:
            return {
                'X': self.X[index],
                'Y': self.y[index]
            }


#%% Set experiment hyperparameters and train, val, test data sets
# Hyperparameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.00001
num_epochs = 100
batch_size = 64

# Create data loaders
train_dataset = PanamaLoadDataset(X_train, y_train, type="Train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn = train_dataset.collate_fn, shuffle=False)

val_dataset = PanamaLoadDataset(X_val, y_val, type="Dev")
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn = val_dataset.collate_fn, shuffle=False)

test_dataset = PanamaLoadDataset(X_test, y_test, type="Test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn = test_dataset.collate_fn, shuffle=False)


#%% Initialize the neural network from Neuromancer blocks

from neuromancer.modules import blocks


block_lstm = blocks.PytorchLSTM(
    insize=input_size, 
    outsize=output_size, 
    hsizes=[hidden_size] * num_layers,
    num_layers=num_layers,
    linear_map=torch.nn.Linear, 
    nonlin=torch.nn.ReLU, 
)

#%% Use Neuromancer to define the Node, objective, constraints and problem

from neuromancer.system import Node
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss


model = block_lstm
model = model.to(device)

lstm_node = Node(model, ['X'],['predicted'], name = 'lstm')

nodes = [lstm_node]

predicted = variable('predicted')
real = variable('Y')

MSE = (real == predicted)^2 # Choose mean squared error as the loss function.
MSE.name='mse'

constraints = []
objectives = [MSE] 

# create constrained optimization loss
loss_ = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(nodes, loss_)

optimizer = torch.optim.Adam(block_lstm.parameters(), lr=0.005) #we choose the Adam optimizer because it reports to have a faster convergence and little memory requirements

logger = BasicLogger(args=None, savedir='test', verbosity=1,
                        stdout=['dev_loss', 'train_loss'])

problem.show()


#%% Initiate Trainer

problem = problem.to(device=device)

N_EPOCHS = 100

trainer = Trainer(
    problem,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    patience=5,
    warmup=30,
    epochs=N_EPOCHS,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
    device=device
)
best_model = trainer.train()
problem.load_state_dict(best_model)


# %% Test the model on test set, show performance metrics and plot forecasts

# Define a function to compute inverse transform of target variable only

def inverse_tranform_y(y_scaled):
    y = y_scaled - scaler.min_[0]
    y = y / scaler.scale_[0]
    return y


print("Testing on test set...")
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    y_pred = model(X_test_tensor)#.numpy()

    y_pred_scaled = inverse_tranform_y(y_pred.cpu())
    y_test_scaled = inverse_tranform_y(y_test)

    # Calculate RMSE
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    mape = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# Visualize predictions against actual data
plt.figure(figsize=(10, 6))
plt.plot(df_test.index[seq_length:], y_test_scaled, label='Actual')
plt.plot(df_test.index[seq_length:], y_pred_scaled, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Electricity Load (MWh)')
plt.title('Panama Electricity Load Forecasting')
plt.grid()
plt.legend()
plt.show()


#%% Randomly choose 72-hour intervals to show forecasts

import random


random.seed(0)
num_samples = 10
forecast_window = 72

start_idxs = random.sample(range(len(y_pred_scaled)), num_samples)

for idx in start_idxs:

    plt.plot(y_test_scaled[idx:idx+forecast_window], label='Real', marker='o')
    plt.plot(y_pred_scaled[idx:idx+forecast_window], label='Predicted', marker='x')

    plt.ylabel('Electricity Load (MWh)')
    plt.xlabel('Time (Hours)')
    plt.grid()
    plt.legend()
    plt.show()


# %%
