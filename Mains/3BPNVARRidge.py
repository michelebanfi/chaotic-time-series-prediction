import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reservoirpy.datasets import lorenz
from RidgeBasedExp.Modules.ESNRidge import ESNReservoir
from RidgeBasedExp.Modules.NVARRidge import NVARReservoir
import pandas as pd

# Function to visualize reservoir states
def plot_reservoir_states(states):
    states = states.cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(states.T, cmap='viridis', cbar=True)
    plt.xlabel('Time Steps')
    plt.ylabel('Reservoir Units')
    plt.title('Reservoir States')
    plt.show()

io_size = 4
degree = 2
ridge_alpha = 1e-2
pred_len = 1
delay = 2 # New parameter
stride = 1  # New parameter

nb_generations = 100
seed_timesteps = 300

nvar = NVARReservoir(io_size, degree, ridge_alpha, delay=delay, stride=stride)

df = pd.read_csv("../../RestrictedThreeBodyProblem/Data/3BP_0.csv")
X = df[['x', 'y', 'vx', 'vy']].values

# scale the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

# Generate target data shifting by pred_len
y = X[:, pred_len:, :]
X = X[:, :-pred_len, :]

# Adjust X and y for stride
X = X[:, ::stride, :]
y = y[:, ::stride, :]

n = int(X.size(1) * 0.8)

X_train1, y_train1 = X[:, :n, :], y[:, :n, :]
X_test1, y_test1 = X[:, n:, :], y[:, n:, :]

# Continue with the generation function, starting with a warmup phase
warming_inputs = X_test1[:, :seed_timesteps, :]
warming_out = nvar(warming_inputs)

# plot the data in 2D separating train and test
fig = plt.figure()
plt.plot(X_train1[0, :, 0].numpy(), X_train1[0, :, 1].numpy(), label='Train')
plt.plot(X_test1[0, :, 0].numpy(), X_test1[0, :, 1].numpy(), label='Test')
plt.plot(warming_inputs[0, :, 0].numpy(), warming_inputs[0, :, 1].numpy(), label='Warmup')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

nvar.fit(X_train1, y_train1)
output = nvar(X_test1)

# Adjust y_test1 for the reduced length due to stride
y_test1_adjusted = y_test1[:, :output.shape[1], :]

# calculate the RMSE
rmse = torch.sqrt(torch.mean((output - y_test1_adjusted) ** 2))
print(f'RMSE: {rmse}')

# calculate the R^2 score
ss_res = torch.sum((y_test1_adjusted - output) ** 2)
ss_tot = torch.sum((y_test1_adjusted - torch.mean(y_test1_adjusted)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R^2: {r2}')

# Plot the results in subplots
fig, axs = plt.subplots(io_size, 1, figsize=(10, 10))
for i in range(io_size):
    axs[i].plot(y_test1_adjusted[0, :, i].numpy(), label='True')
    axs[i].plot(output[0, :, i].detach().numpy(), label='Predicted')
    axs[i].legend()
plt.show()

print("Wout shape:", nvar.Wout.shape)

# the generation can be performed only for pred_len = 1.
if pred_len == 1:
    X_gen = np.zeros((nb_generations, io_size))
    y = warming_inputs[:, -delay:, :].squeeze(0).detach().numpy()

    print("warming_inputs shape:", warming_inputs.shape)
    print("y shape:", y.shape)
    print("X_gen shape:", X_gen.shape)

    for t in range(nb_generations):
        input = torch.tensor(y, dtype=torch.float32)
        if input.dim() == 2:
            input = input.unsqueeze(0)  # Add batch dimension if needed
        output = nvar(input)
        y = np.roll(y, -1, axis=0)
        y[-1] = output[0, -1, :].detach().numpy()
        X_gen[t] = y[-1]  # This should now work correctly

    X_t = X_test1[:, seed_timesteps:seed_timesteps + nb_generations * stride:stride]

    # plot the 4 separate variables in 4 subplots
    fig, axs = plt.subplots(io_size, 1, figsize=(10, 10))
    for i in range(io_size):
        axs[i].plot(X_t[0, :, i].numpy(), label='True')
        axs[i].plot(X_gen[:, i], label='Generated')
        axs[i].legend()
    plt.show()

    # plot the data in 2D
    fig = plt.figure()
    plt.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    plt.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    plt.plot(warming_inputs[0, :, 0].numpy(), warming_inputs[0, :, 1].numpy(), label='Warmup')
    plt.xlabel('X')
    plt.ylabel('Y')
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