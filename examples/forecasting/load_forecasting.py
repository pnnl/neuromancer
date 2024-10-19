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

# ElectricityLoadDiagrams20112014 from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
response = requests.get(url, stream=True)


#%% Extract the zip file

z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

#%% Read as pandas dataframe and set index

df = pd.read_csv("LD2011_2014.txt", delimiter=";", decimal=',')
df = df.rename(columns={"Unnamed: 0": "date"}).set_index("date")
df.index = pd.to_datetime(df.index)
df


#%% Skip 2011 since some clients are recorded from 2012

df = df[df.index > "2012-01-01 00:00:00"] 
df


#%% Choose a client (1-370) to train and test forecasting as dataset contains 

CLIENT = 1
suffix = (3 - len(str(CLIENT))) * "0" + str(CLIENT)  # determine num zeroes before id 
df_client = df[f"MT_{suffix}"]
df_client = pd.DataFrame(df_client)
df_client

#%%

# Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_client)

# Create sequences and labels for training
seq_length = 4 * 6 # lookback (last 6 hours with 15 min. intervals)
X, y = [], []
for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i+seq_length])
    y.append(df_scaled[i+seq_length])

X, y = np.array(X), np.array(y)


#%%
# Split the data into training and test sets

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create a custom dataset class for PyTorch DataLoader
class UCIElectricityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


#%%
# Define the LSTM model

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=16):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hiddens, _) = self.lstm(x)
        out = self.linear(hiddens[-1])
        return out


#%%
# Hyperparameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 1
output_size = 1
learning_rate = 0.001
num_epochs = 50
batch_size = 64
device = "cpu"

# Create data loaders
train_dataset = UCIElectricityDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


#%%
# Initialize the model, loss function, and optimizer

lstm_torch = LSTMModel(
    input_size=1, 
    output_size=1, 
    num_layers=num_layers, 
    hidden_size=hidden_size)


#%% Initialize Neuromancer blocks

from neuromancer.modules import blocks
from neuromancer.modules import activations
from neuromancer import slim


block_rnn = blocks.RNN(
    insize=1, 
    outsize=1,
    bias=True,
    linear_map=slim.linear.SVDLinear,
    nonlin=activations.BLU,
    hsizes=[hidden_size] * 1)

block_rnn_torch = blocks.PytorchRNN(
    insize=1,
    outsize=1,
    hsizes=[hidden_size]*1
)

block_lstm = blocks.PytorchLSTM(
    insize=1, 
    outsize=1, 
    hsizes=[hidden_size] * 1,
    num_layers=num_layers
)


#%% Train and evaluate the selected model

model = block_rnn
model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

EVAL_PERIOD = 5

# Training the model
for epoch in range(num_epochs):

    with tqdm(train_loader, unit="batch") as tepoch: # show progress bar for iterations
        for idx, batch in enumerate(tepoch):
            
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

                y_pred_scaled = scaler.inverse_transform(y_pred)
                y_test_scaled = scaler.inverse_transform(y_test)

                # Calculate RMSE
                mse = mean_squared_error(y_test_scaled, y_pred_scaled)
                rmse = np.sqrt(mse)
                print(f"Root Mean Squared Error (RMSE): {rmse}")
                mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
                print(f"Mean Absolute Error (MAE): {mae:.2f}")


#%%
# Visualize predictions against actual data
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+seq_length:], y_test_scaled, label='Actual')
plt.plot(df.index[train_size+seq_length:], y_pred_scaled, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption (kW)')
plt.title('Electricity Consumption Forecasting')
plt.grid()
plt.legend()
plt.show()

# %%
