#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from tqdm import tqdm


#%% Set numpy and torch seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


#%%
# Sample data for Electricity Consumption 

N_SAMPLES = 1000
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='h'),
    'consumption': np.random.randint(100, 1000, N_SAMPLES)
}
df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

#%%

# Preprocessing
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create sequences and labels for training
seq_length = 24 # lookback
X, y = [], []
for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i + seq_length])
    y.append(df_scaled[i + seq_length])

X, y = np.array(X), np.array(y)

#%%
# Split the data into training and test sets

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create a custom dataset class for PyTorch DataLoader
class TimeSeriesDataset(Dataset):
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
train_dataset = TimeSeriesDataset(X_train, y_train)
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

model = block_lstm
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
                mape = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled) * 100
                print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


#%%
# Visualize predictions against actual data
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+seq_length:], y_test_scaled, label='Actual')
plt.plot(df.index[train_size+seq_length:], y_pred_scaled, label='Predicted')
plt.xlabel('Timestamp')
plt.ylabel('Electricity Consumption')
plt.title('Electricity Consumption Prediction using LSTM (PyTorch)')
plt.legend()
plt.show()

# %%
