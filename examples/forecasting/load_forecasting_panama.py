#%% Load forecasting on UCI Electricity Load Diagrams 2011-2014 dataset (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
# Downloads and preproceses the data. Then, trains RNN or LSTM model from Neuromancer blocks and plots the load forecast.

#%% Imports

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tqdm import tqdm
import zipfile


#%% Set numpy and torch seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


#%% Download data set

# Short-term electricity load forecasting (Panama case study) from Mendeley Data 
# (DOI:10.17632/byx7sztj59.1)
url = "https://data.mendeley.com/public-files/datasets/byx7sztj59/files/94ef3285-7ce4-43e2-9fd3-9cb3f592b89c/file_downloaded"
response = requests.get(url, stream=True)

file_name = "continuous_dataset.csv"
with open(file_name, "wb") as file:
    file.write(response.content)

#%% Read csv file as Pandas dataframe

df = pd.read_csv(file_name)
df


#%% Read as pandas dataframe and set index

df = df.set_index("datetime")
df.index = pd.to_datetime(df.index)
df


#%% Encode time information using Radial Basis Functions (RBFs)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklego.preprocessing import RepeatingBasisFunction


df["day_of_year"] = df.index.day_of_year
rbf = RepeatingBasisFunction(n_periods=12,
                         	column="day_of_year",
                         	input_range=(1,365),
                         	remainder="drop")
rbf.fit(df)
df_rbf = pd.DataFrame(
    index=df.index,
    data=rbf.transform(df),
    columns=[f'rbf_{i}' for i in range(12)]
)

df_rbf.plot(subplots=True, figsize=(14, 8),
     	sharex=True, title="Radial Basis Functions",
     	legend=False)

#%% Concat dfs

df = pd.concat([df, df_rbf], axis=1)
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

excluded_cols = ['Holiday_ID', 'day_of_year']
excluded_cols.extend(df_rbf.columns)
df = df.drop(columns=excluded_cols)



#%%
# Split the data into training and test sets

seq_length = 72 # lookback = 72 hours
train_size = int(0.8 * len(df))
df_train = df[:train_size]
df_test = df[train_size + seq_length:] # add gap between train and test datasets to prevent data leakage:]


#%%
# Scale the data
scaler = MinMaxScaler(feature_range=(-1,1))
train_data = scaler.fit_transform(df_train)
train_data

#%% Transform test data using the scaler parameters for train data

test_data = scaler.transform(df_test)
test_data

#%% Define a function to compute inverse transform of target variable only

def inverse_tranform_y(y_scaled):
    y = y_scaled - scaler.min_[0]
    y = y / scaler.scale_[0]
    return y


#%%
# Create sequential data with a window size of 72

def create_timeseries(data, len_sequence):
    X, y = [], []
    for i in range(len(data) - len_sequence):
        X.append(data[i:i+len_sequence])
        y.append(data[i+len_sequence][0])

    X, y = np.array(X), np.array(y).reshape(-1,1)
    return X, y

X_train, y_train = create_timeseries(train_data, seq_length)
X_test, y_test = create_timeseries(test_data, seq_length)


#%%

# Create a custom dataset class for PyTorch DataLoader
class PanamaLoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


#%%
# Hyperparameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.00001
num_epochs = 100
batch_size = 64
device = "cuda"

# Create data loaders
train_dataset = PanamaLoadDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


#%% Initialize Neuromancer blocks

from neuromancer.modules import blocks
from neuromancer.modules import activations
from neuromancer import slim


block_rnn = blocks.RNN(
    insize=input_size, 
    outsize=output_size, 
    bias=True,
    linear_map=slim.linear.SVDLinear,
    nonlin=activations.BLU,
    hsizes=[hidden_size] * 1)

block_lstm = blocks.PytorchLSTM(
    insize=input_size, 
    outsize=output_size, 
    hsizes=[hidden_size] * num_layers,
    num_layers=num_layers,
    linear_map=torch.nn.Linear, 
    nonlin=torch.nn.ReLU, 
)


#%% Train and evaluate the selected model

model = block_lstm
model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

EVAL_PERIOD = 10

# Training the model
for epoch in range(num_epochs):

    with tqdm(train_loader, unit="batch") as tepoch: # show progress bar for iterations
        for idx, batch in enumerate(tepoch):
            model.train()
            
            tepoch.set_description(f"Epoch {epoch}")

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {loss.item():.4f}')
        
        # Evaluate model
        if epoch % EVAL_PERIOD == EVAL_PERIOD - 1:
            model.eval()
            print("Validating...")
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

                y_pred = model(X_test_tensor)#.numpy()
                loss_val = criterion(y_pred, y_test_tensor)
                print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {loss.item():.4f}, Validation loss: {loss_val.item():.4f}')

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


#%%
# Visualize predictions against actual data
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+seq_length*2:], y_test_scaled, label='Actual')
plt.plot(df.index[train_size+seq_length*2:], y_pred_scaled, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption (kW)')
plt.title('Electricity Consumption Forecasting')
plt.grid()
plt.legend()
plt.show()


# %%
