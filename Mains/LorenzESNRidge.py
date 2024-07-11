import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Reservoirs.ESNRidge import ESNReservoir

# Function to visualize reservoir states
def plot_reservoir_states(states):
    states = states.cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(states.T, cmap='viridis', cbar=True)
    plt.xlabel('Time Steps')
    plt.ylabel('Reservoir Units')
    plt.title('Reservoir States')
    plt.show()

# ESN params
io_size = 3
reservoir_size = 1024
pred_len = 1
spectral_radius = 0.9
sparsity = 0.1
leaking_rate = 0.3
connectivity = 0.1
ridge_alpha = 0.01

nb_generations = 50
seed_timesteps = 100

esn = ESNReservoir(io_size, reservoir_size, pred_len, spectral_radius=spectral_radius, sparsity=sparsity,
                   leaking_rate=leaking_rate, connectivity=connectivity, ridge_alpha=ridge_alpha)

X = pd.read_csv("../Data/Lorenz/lorenz_0.csv")
X = X[['x', 'y', 'z']].values
X = X[::20]

# plot the data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
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
fig, axs = plt.subplots(io_size, 1, figsize=(10, 10))
for i in range(io_size):
    axs[i].plot(y_test1[0, :, i].numpy(), label='True')
    axs[i].plot(output[0, :, i].detach().numpy(), label='Predicted')
    axs[i].legend()
plt.show()

# After fitting the model. Lol _Wout Van Aert_ is a Belgian cyclist
print("Wout shape:", esn.Wout.shape)

# the generation can be performed only for pred_len = 1.
# This is because the model is trained to predict only one step ahead
if pred_len == 1:

    # Continue with the generation function, starting with a warmup phase
    warming_inputs = X_test1[:, :seed_timesteps, :]
    _, h = esn(warming_inputs)

    X_gen = np.zeros((nb_generations, io_size))
    y = warming_inputs[:, -1, :].detach().numpy()
    for t in range(nb_generations):
        input = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        output, h = esn(input, h)
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

    print("Prediction length must be 1 to generate data")