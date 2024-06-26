import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time
from Utils.DataEvaluator import evaluate
from Reservoirs.LSTMReservoir import LSTMReservoir
from Utils.DataLoader import loadData
from Benchmarks.LSTM import LSTM
import matplotlib.pyplot as plt

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV
df = pd.read_csv('Data/3BP.csv')
data = df[['x', 'y', 'vx', 'vy']].values
t = df['time'].values

# Define sequence length
seq_len = 1

# Define the model parameters
input_size = 4
reservoir_size = 128
output_size = 4
batch_size = 10
num_epochs = 10

# Load the data
train_dataloader, val_sequences_torch, val_targets_torch, val_t = loadData(data, t, seq_len, batch_size=batch_size)

# use the Reservoir
model = LSTMReservoir(input_size, reservoir_size, output_size, seq_len=seq_len)
modelBenchmark = LSTM(input_size, reservoir_size, output_size, seq_len=seq_len)


# Define training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# start counting the time
start = time.time()
# Train the model
val_predictions_np, val_target_np, losses = (
    evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_sequences_torch, val_targets_torch))

# stop counting the time
end = time.time()
print('Time elapsed: ', end - start)

optimizer = optim.Adam(modelBenchmark.parameters(), lr=0.001)
criterion = nn.MSELoss()
# start counting the time
start = time.time()
# Train the benchmark model
val_predictions_np_benchmark, val_target_np_benchmark, lossesBenchmark = (
    evaluate(num_epochs, criterion, optimizer, modelBenchmark, train_dataloader, val_sequences_torch, val_targets_torch))

# stop counting the time
end = time.time()
print('Time elapsed: ', end - start)


# Plotting the predictions
plt.figure(figsize=(15, 5))

# Plotting the predictions
for i, var_name in enumerate(['x', 'y']):
    plt.subplot(1, 2, i + 1)
    plt.plot(val_t[seq_len:], val_target_np[:, :, i], label='Real')
    plt.plot(val_t[seq_len:], val_predictions_np[:, :, i], label='Predicted (Reservoir)')
    plt.plot(val_t[seq_len:], val_predictions_np_benchmark[:, :, i], label='Predicted (Benchmark)')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.grid()
plt.savefig('Media/3BP_prediction.png')
plt.close()

