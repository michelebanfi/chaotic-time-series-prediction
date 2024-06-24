import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Reservoirs.ESNReservoir import ESNReservoir
from Reservoirs.LSTMReservoir import LSTMReservoir
from Reservoirs.GRUReservoir import GRUReservoir
from Utils.DataLoader import loadData
from Utils.DataEvaluator import evaluate

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV
df = pd.read_csv('lorenz_data.csv')
data = df[['x', 'y', 'z']].values
t = df['time'].values

# Define sequence length
seq_len = 2

# Define the model parameters
input_size = 3
reservoir_size = 100
output_size = 3

# Load the data
train_dataloader, val_sequences_torch, val_targets_torch, val_t = loadData(data, t, seq_len)

# use the Reservoir
model = GRUReservoir(input_size, reservoir_size, output_size, seq_len=seq_len)

# Define training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 1 # it will break with 0 epochs

# Train the model
val_predictions_np, val_target_np, moel, losses = evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_sequences_torch, val_targets_torch)

# Plotting the predictions
plt.figure(figsize=(15, 5))

stepToShow = min(seq_len - 1, 4)
for i, var_name in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, i+1)
    plt.plot(val_t[seq_len:], val_target_np[:, stepToShow, i], label='True')
    plt.plot(val_t[seq_len:], val_predictions_np[:, stepToShow, i], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.savefig('Media/lorenz_predictions.png')
plt.show()
plt.close()

# plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('Media/lorenz_loss.png')
plt.show()
plt.close()
