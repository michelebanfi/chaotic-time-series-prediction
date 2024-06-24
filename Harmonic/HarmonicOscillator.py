import pandas as pd
import torch
from Reservoirs.LSTMReservoir import LSTMReservoir
from Reservoirs.ESNReservoir import ESNReservoir
from Benchmarks.LSTM import LSTM
from Utils.DataLoader import loadData
from Utils.DataEvaluator import evaluate

# load the data
df = pd.read_csv('harmonic_data.csv')
data = df[['x', 'v']].values
t = df['time'].values

seq_len = 1

input_size = 2
reservoir_size = 100
output_size = 2

train_dataloader, val_sequences_torch, val_targets_torch, val_t = loadData(data, t, seq_len)

# use the Reservoir
model = LSTM(input_size, reservoir_size, output_size, seq_len=seq_len)

# Define training parameters
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 2

# Train the model
model, val_predictions_np, val_target_np, losses = evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_sequences_torch, val_targets_torch)
