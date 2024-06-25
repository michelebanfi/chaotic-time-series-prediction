import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Reservoirs.GRUReservoir import GRUReservoir
from Reservoirs.LSTMReservoir import LSTMReservoir
from Benchmarks.LSTM import LSTM
from Benchmarks.GRU import GRU
from Utils.DataLoader import loadData
from Utils.DataEvaluator import evaluate
import time

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV
df = pd.read_csv('Data/lorenz_data.csv')
data = df[['x', 'y', 'z']].values
t = df['time'].values

# Define sequence length
seq_len = 1

# Define the model parameters
input_size = 3
reservoir_size = 128
output_size = 3
batch_size = 1

# Load the data
train_dataloader, val_sequences_torch, val_targets_torch, val_t = loadData(data, t, seq_len, batch_size=batch_size)

# use the Reservoir
model = LSTMReservoir(input_size, reservoir_size, output_size, seq_len=seq_len)
modelBenchmark = LSTM(input_size, reservoir_size, output_size, seq_len=seq_len)

# Define training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1 # it will break with 0 epochs

lurido = val_sequences_torch
zozzo = val_targets_torch

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
    evaluate(num_epochs, criterion, optimizer, model, train_dataloader, lurido, zozzo))

# stop counting the time
end = time.time()
print('Time elapsed: ', end - start)

# Plotting the predictions
plt.figure(figsize=(15, 5))

stepToShow = min(seq_len, 100)
for i, var_name in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 4, i+1)
    plt.plot(val_t[stepToShow:], val_target_np[:, stepToShow - 1, i], label='True')
    plt.plot(val_t[stepToShow:], val_predictions_np[:, stepToShow - 1, i], label='Predicted (Reservoir)')
    plt.plot(val_t[stepToShow:], val_predictions_np_benchmark[:, stepToShow - 1, i], label='Predicted (Benchmark)')
    # plot the residuals between the predicted and the true values
    plt.plot(val_t[stepToShow:], val_target_np[:, stepToShow - 1, i] - val_predictions_np[:, stepToShow - 1, i], label='Residuals (Reservoir)')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.savefig('Media/lorenz_predictions.png')
#plt.show()
plt.close()

# plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('Media/lorenz_loss.png')
#plt.show()
plt.close()

# plot the loss
plt.plot(lossesBenchmark)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('Media/lorenz_loss.png')
#plt.show()
plt.close()
