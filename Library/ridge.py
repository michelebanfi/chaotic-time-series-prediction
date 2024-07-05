import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge, Lasso
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from reservoirpy.datasets import lorenz

# set torch seed
torch.manual_seed(0)

class ESNReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, pred_len, spectral_radius=0.90, sparsity=0.1,
                 ridge_alpha=0.03, leaking_rate=1.0, connectivity=0.1):
        super(ESNReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.pred_len = pred_len
        self.ridge_alpha = ridge_alpha
        self.leaking_rate = leaking_rate

        # Input weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size) * 0.1, requires_grad=False)

        # Reservoir weights
        W = torch.randn(reservoir_size, reservoir_size)

        # Apply sparsity and connectivity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        W *= mask

        # Adjust spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        W *= spectral_radius / max_eigenvalue

        # Apply connectivity
        conn_mask = (torch.rand(reservoir_size, reservoir_size) < connectivity).float()
        W *= conn_mask

        self.W = nn.Parameter(W, requires_grad=False)

        # Placeholder for ridge regression weights
        self.Wout = None
        self.Wout_bias = None

    def forward(self, x, h=None):
        device = x.device
        if h is None:
            h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = x.size(1)

        states = []
        for t in range(input_len):
            input = x[0, t, :].unsqueeze(0)
            h_new = F.tanh(self.Win @ input.T + self.W @ h.T).T
            h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
            states.append(h)

        states = torch.cat(states, dim=0)
        if self.Wout is not None:
            outputs = torch.matmul(states, self.Wout.T) + self.Wout_bias
        else:
            outputs = torch.zeros(states.size(0), self.output_size).to(device)

        return outputs.unsqueeze(0), h

    def fit(self, X, y):
        device = X.device
        h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = X.size(1)
        states = []

        with torch.no_grad():
            for t in range(input_len):
                input = X[0, t, :].unsqueeze(0)
                h_new = F.tanh(self.Win @ input.T + self.W @ h.T).T
                h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
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
input_size = 2
reservoir_size = 512
output_size = 2
pred_len = 1
nb_generations = 300
seed_timesteps = 300


esn = ESNReservoir(input_size, reservoir_size, output_size, pred_len, spectral_radius=0.9, sparsity=0.1, leaking_rate=0.3, connectivity=0.1, ridge_alpha=0.03)

# X = lorenz(10000)

df = pd.read_csv("../RestrictedThreeBodyProblem/Data/3BP_0.csv")
variables = ['x', 'y']
X = df[variables].values

# plot the data in 2D - 3BP
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X[:, 0], X[:, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
#plt.savefig('3bp.png')
plt.show()

# scale the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

n = int(X.size(1) * 0.8)

# Generate target data shifting by pred_len
y = X[:, pred_len:, :]
X = X[:, :-pred_len, :]

X_train1, y_train1 = X[:, :n, :], y[:, :n, :]
X_test1, y_test1 = X[:, n:, :], y[:, n:, :]

esn.fit(X_train1, y_train1)
output, _ = esn(X_test1)

# calculate the RMSE
rmse = torch.sqrt(torch.mean((output - y_test1) ** 2))
print(f'RMSE: {rmse}')

# calculate the R^2 score
ss_res = torch.sum((y_test1 - output) ** 2)
ss_tot = torch.sum((y_test1 - torch.mean(y_test1)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R^2: {r2}')

# Plot the results in subplots
fig, axs = plt.subplots(input_size, 1, figsize=(10, 10))
for i in range(input_size):
    axs[i].plot(y_test1[0, :, i].numpy(), label='True')
    axs[i].plot(output[0, :, i].detach().numpy(), label='Predicted')
    axs[i].legend()
plt.savefig('lorenz.png')
#plt.close()
plt.show()

# After fitting the model
print("Wout shape:", esn.Wout.shape)

if pred_len == 1:
    # Continue with the generation function, starting with a warmup phase
    warming_inputs = X_test1[:, :seed_timesteps, :]
    _, h = esn(warming_inputs)

    X_gen = np.zeros((nb_generations, input_size))
    y = warming_inputs[:, -1, :].detach().numpy()
    for t in range(nb_generations):
        input = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        output, h = esn(input, h)
        y = output[:, 0, :]
        z = output[0, 0, :].detach().numpy()
        X_gen[t] = z

    X_t = X_test1[:, seed_timesteps: nb_generations + seed_timesteps]

    # plot the 3 separate variables in 3 subplots
    fig, axs = plt.subplots(input_size, 1, figsize=(10, 10))
    for i in range(input_size):
        axs[i].plot(X_t[0, :, i].numpy(), label='True')
        axs[i].plot(X_gen[:, i], label='Generated')
        axs[i].legend()
    plt.savefig('lorenz_gen.png')
    plt.close()
    plt.show()

    # plot the data in 3D - LORENZ
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    # ax.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    # ax.plot(warming_inputs[0, :, 0], warming_inputs[0, :, 1], label='Warming')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.legend()
    # plt.savefig('lorenz_gen_3d.png')
    # plt.close()
    # # plt.show()

    # plot the data in 2D - 3BP
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    ax.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    ax.plot(warming_inputs[0, :, 0], warming_inputs[0, :, 1], label='Warming')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.legend()
    # plt.savefig('3bp_gen.png')
    # plt.close()
    plt.show()

    X_t = X_t.squeeze(0).numpy()

    # calculate the RMSE
    rmse = np.sqrt(np.mean((X_gen - X_t) ** 2))
    print(f'RMSE: {rmse}')

    # calculate the R^2 score
    ss_res = np.sum((X_t - X_gen) ** 2)
    ss_tot = np.sum((X_t - np.mean(X_t)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f'R^2: {r2}')

else:
    print("Prediction length must be 1 to generate data")