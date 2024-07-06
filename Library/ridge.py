import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class ESNReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, pred_len, spectral_radius=1.367163082884632, sparsity=0.1,
                 ridge_alpha=0.06277358548329898):
        super(ESNReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.pred_len = pred_len
        self.ridge_alpha = ridge_alpha

        # Input weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size), requires_grad=False)

        # Reservoir weights
        self.W = nn.Parameter(torch.randn(reservoir_size, reservoir_size), requires_grad=False)

        # Adjust spectral radius
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W.data)))

        # Apply sparsity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        self.W.data *= mask

        # Placeholder for ridge regression weights
        self.Wout = None

    def forward(self, x):
        device = x.device
        h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = x.size(1)

        states = []
        for t in range(self.pred_len):
            input = x[0, t, :].unsqueeze(0)
            h = F.leaky_relu(self.Win @ input.T + self.W @ h.T).T
            states.append(h)

        states = torch.cat(states, dim=0)
        if self.Wout is not None:
            outputs = torch.matmul(states, self.Wout.T)
        else:
            outputs = torch.zeros(states.size(0), self.output_size).to(device)

        return outputs.unsqueeze(0)

    def fit(self, X, y):
        device = X.device
        h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = X.size(1)
        states = []

        with torch.no_grad():
            for t in range(input_len):
                input = X[0, t, :].unsqueeze(0)
                h = F.leaky_relu(self.Win @ input.T + self.W @ h.T).T
                states.append(h)

        states = torch.cat(states, dim=0).cpu().numpy()

        y = y.cpu().numpy()

        y = y.squeeze(0)

        # Perform ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(states, y)
        self.Wout = torch.tensor(ridge.coef_, dtype=torch.float32).to(device)
        self.Wout_bias = torch.tensor(ridge.intercept_, dtype=torch.float32).to(device)

# Function to visualize reservoir states
def plot_reservoir_states(states):
    states = states.cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(states.T, cmap='viridis', cbar=True)
    plt.xlabel('Time Steps')
    plt.ylabel('Reservoir Units')
    plt.title('Reservoir States')
    plt.show()

# Usage
input_size = 3
reservoir_size = 512
output_size = 3
pred_len = 1
nb_generations = 100

esn = ESNReservoir(input_size, reservoir_size, output_size, pred_len)

df = pd.read_csv("../Lorenz/Data/lorenz_0.csv")
variables = ['x', 'y', 'z']
X = df[variables].values

# 3D plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(X[:, 0], X[:, 1], X[:, 2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[variables].values)
X = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

# convert to tensor
#X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

n = int(X.size(1) * 0.8)

# Generate target data shifting by pred_len
y = X[:, pred_len:, :]
X = X[:, :-pred_len, :]

X_train1, y_train1 = X[:, :n, :], y[:, :n, :]
X_test1, y_test1 = X[:, n:, :], y[:, n:, :]

esn.fit(X_train1, y_train1)
output = esn(X_test1)

# calculate the RMSE
rmse = torch.sqrt(torch.mean((output - y_test1) ** 2))
print(f'RMSE: {rmse}')

# calculate the R^2 score
ss_res = torch.sum((y_test1 - output) ** 2)
ss_tot = torch.sum((y_test1 - torch.mean(y_test1)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R^2: {r2}')

# Plot the results in subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i in range(3):
    axs[i].plot(y_test1[0, :, i].numpy(), label='True')
    axs[i].plot(output[0, :, i].detach().numpy(), label='Predicted')
    axs[i].legend()
plt.show()

# After fitting the model
print("Wout shape:", esn.Wout.shape)

# Visualize regression weights
# plt.figure(figsize=(12, 6))
# sns.heatmap(esn.Wout.cpu().detach().numpy(), cmap='coolwarm', cbar=True)
# plt.title('Regression Weights (Wout)')
# plt.show()

# Continue with the generation function, starting with a warmup phase
seed_timesteps = 100
warming_inputs = X_test1[:, :seed_timesteps, :]
warming_out = esn(warming_inputs)

X_gen = np.zeros((nb_generations, 3))
y = warming_out[:, -1, :].detach().numpy()
for t in range(nb_generations):
    input = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    output = esn(input)
    y = output[:, 0, :]
    z = output[0, 0, :].detach().numpy()
    X_gen[t] = z

X_t = X_test1[:, seed_timesteps: nb_generations + seed_timesteps]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_gen[:, 0], X_gen[:, 1], X_gen[:, 2], 'r')
ax.plot(X_t[:, :, 0], X_t[:, :, 1], X_t[:, :, 2], 'b')
ax.plot(warming_inputs[:, :, 0], warming_inputs[:, :, 1], warming_inputs[:, :, 2], 'g')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(['Generated', 'True', 'Warmup'])
plt.show()