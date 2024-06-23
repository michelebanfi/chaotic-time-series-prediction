import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV
df = pd.read_csv('lorenz_data.csv')
data = df[['x', 'y', 'z']].values
t = df['time'].values

print("loading data")

# Prepare the data for PyTorch
data_torch = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Adding batch dimension
target_torch = data_torch[:, 1:, :]  # Target is the next time step

# Create DataLoader for batching
dataset = torch.utils.data.TensorDataset(data_torch[:, :-1, :], target_torch)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

print("data loaded")

# Define the reservoir model
class Reservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1):
        super(Reservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Input weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size))

        # Reservoir weights
        self.W = nn.Parameter(torch.randn(reservoir_size, reservoir_size))

        # Output weights
        self.Wout = nn.Linear(reservoir_size, output_size)

        # Adjust spectral radius
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W.data)))

        # Apply sparsity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        self.W.data *= mask

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_size)
        outputs = []

        for t in range(seq_len):
            h = torch.tanh(self.Win @ x[:, t, :].T + self.W @ h.T).T
            outputs.append(self.Wout(h))

        outputs = torch.stack(outputs, dim=1)
        return outputs


# Define the model parameters
input_size = 3
reservoir_size = 100
output_size = 3

print("creating model")

# Create the reservoir model
model = Reservoir(input_size, reservoir_size, output_size)

print("model created")

# Print the model architecture
print(model)

# Define training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(data_torch[:, :-1, :])
    rmse = torch.sqrt(criterion(predictions, target_torch))
    predictions_np = predictions.squeeze(0).numpy()
    target_np = target_torch.squeeze(0).numpy()
    print(f'RMSE: {rmse.item():.4f}')


# Check dimensions
print("Shape of t:", t[1:].shape)
print("Shape of target_np:", target_np.shape)
print("Shape of predictions_np:", predictions_np.shape)

# Plotting the predictions
plt.figure(figsize=(15, 5))

# Plot each state variable separately
for i, var_name in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, i+1)
    plt.plot(t[1:], target_np[:, i], label='True')
    plt.plot(t[1:], predictions_np[:, i], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.savefig('lorenz_predictions.png')
plt.show()