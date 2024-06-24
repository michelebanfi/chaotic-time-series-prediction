import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from Reservoirs.ESNReservoir import ESNReservoir
from Reservoirs.LSTMReservoir import LSTMReservoir

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV
df = pd.read_csv('lorenz_data.csv')
data = df[['x', 'y', 'z']].values
t = df['time'].values

print("loading data")

# # Split the data into training and validation sets
# train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
# train_t, val_t = train_test_split(t, test_size=0.2, shuffle=False)
#
# # Function to create sequences
# def create_sequences(data, seq_len):
#     sequences = []
#     targets = []
#     for i in range(len(data) - seq_len):
#         sequences.append(data[i:i + seq_len])
#         targets.append(data[i + 1:i + seq_len + 1])
#     return np.array(sequences), np.array(targets)
#
# # Define sequence length
# seq_len = 1
#
# # Create sequences for training and validation
# train_sequences, train_targets = create_sequences(train_data, seq_len)
# val_sequences, val_targets = create_sequences(val_data, seq_len)
#
# # Convert to PyTorch tensors
# train_sequences_torch = torch.tensor(train_sequences, dtype=torch.float32)
# train_targets_torch = torch.tensor(train_targets, dtype=torch.float32)
# val_sequences_torch = torch.tensor(val_sequences, dtype=torch.float32)
# val_targets_torch = torch.tensor(val_targets, dtype=torch.float32)
#
# # Create DataLoader for batching
# train_dataset = torch.utils.data.TensorDataset(train_sequences_torch, train_targets_torch)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
train_t, val_t = train_test_split(t, test_size=0.2, shuffle=False)

# Prepare the data for PyTorch
train_data_torch = torch.tensor(train_data, dtype=torch.float32).unsqueeze(0)  # Adding batch dimension
train_target_torch = train_data_torch[:, 1:, :]  # Target is the next time step

val_data_torch = torch.tensor(val_data, dtype=torch.float32).unsqueeze(0)  # Adding batch dimension
val_targets_torch = val_data_torch[:, 1:, :]  # Target is the next time step

# Create DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(train_data_torch[:, :-1, :], train_target_torch)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

print("Data loaded")

# Define the model parameters
input_size = 3
reservoir_size = 100
output_size = 3

# use the ESNReservoir
model = LSTMReservoir(input_size, reservoir_size, output_size)

# Define training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

losses = []
accuracies = []

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    val_predictions = model(val_data_torch[:, :-1, :])
    rmse = torch.sqrt(criterion(val_predictions, val_targets_torch))
    val_predictions_np = val_predictions.squeeze(0).numpy()
    val_target_np = val_targets_torch.squeeze(0).numpy()
    print(f'RMSE: {rmse.item():.4f}')


# Check dimensions
print("Shape of t:", t[1:].shape)
print("Shape of target_np:", val_target_np.shape)
print("Shape of predictions_np:", val_predictions_np.shape)

# Plotting the predictions
plt.figure(figsize=(15, 5))

# Plot each state variable separately
for i, var_name in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, i+1)
    plt.plot(val_t[1:], val_target_np[:, i], label='True')
    plt.plot(val_t[1:], val_predictions_np[:, i], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.savefig('lorenz_predictions.png')
plt.show()
plt.close()

# plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('lorenz_loss.png')
plt.show()
plt.close()
