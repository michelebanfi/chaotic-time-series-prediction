import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from reservoirpy.datasets import lorenz
from Library.Modules.NVARRidge import NVARReservoir

torch.manual_seed(0)

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
io_size = 3
degree = 2
ridge_alpha = 0.1
pred_len = 1

nb_generations = 100
seed_timesteps = 100

nvar = NVARReservoir(io_size, degree, ridge_alpha)

timesteps = 10000

X = lorenz(timesteps)

# plot the data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

n = int(X.size(1) * 0.8)

# Generate target data shifting by pred_len
y = X[:, pred_len:, :]
X = X[:, :-pred_len, :]

X_train1, y_train1 = X[:, :n, :], y[:, :n, :]
X_test1, y_test1 = X[:, n:, :], y[:, n:, :]

nvar.fit(X_train1, y_train1)
output = nvar(X_test1)

# calculate the RMSE
rmse = torch.sqrt(torch.mean((output - y_test1) ** 2))
print(f'RMSE: {rmse}')

# calculate the R^2 score
ss_res = torch.sum((y_test1 - output) ** 2)
ss_tot = torch.sum((y_test1 - torch.mean(y_test1)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R^2: {r2}')

# Plot the results in subplots
fig, axs = plt.subplots(io_size, 1, figsize=(10, 10))
for i in range(io_size):
    axs[i].plot(y_test1[0, :, i].numpy(), label='True')
    axs[i].plot(output[0, :, i].detach().numpy(), label='Predicted')
    axs[i].legend()
plt.show()

# After fitting the model
print("Wout shape:", nvar.Wout.shape)

if pred_len == 1:
    # Continue with the generation function, starting with a warmup phase
    warming_inputs = X_test1[:, :seed_timesteps, :]
    warming_out = nvar(warming_inputs)

    X_gen = np.zeros((nb_generations, 3))
    y = warming_out[:, -1, :].detach().numpy()

    for t in range(nb_generations):
        input = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        output = nvar(input)
        y = output[:, 0, :]
        z = output[0, 0, :].detach().numpy()
        X_gen[t] = z

    X_t = X_test1[:, seed_timesteps: nb_generations + seed_timesteps]

    # plot the 3 separate variables in 3 subplots
    fig, axs = plt.subplots(io_size, 1, figsize=(10, 10))
    for i in range(io_size):
        axs[i].plot(X_t[0, :, i].numpy(), label='True')
        axs[i].plot(X_gen[:, i], label='Generated')
        axs[i].legend()
    plt.show()

    # plot the data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    ax.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    ax.plot(warming_inputs[0, :, 0], warming_inputs[0, :, 1], label='Warming')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
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
    print("Generation can be performed only for pred_len = 1")


