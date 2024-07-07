import pandas as pd
import numpy as np
import torch
from RidgeBasedExp.Modules.ESNRidge import ESNReservoir
import matplotlib.pyplot as plt

df = pd.read_csv("../../RestrictedThreeBodyProblem/Data/3BP_0.csv")
X = df[['x', 'y', 'vx', 'vy']].values

# scale the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

n = int(X.size(1) * 0.8)

pred_len = 1
io_size = 4
nb_generations = 100
seed_timesteps = 100

y = X[:, pred_len:, :]
X = X[:, :-pred_len, :]

X_train1, y_train1 = X[:, :n, :], y[:, :n, :]
X_test1, y_test1 = X[:, n:, :], y[:, n:, :]

warming_inputs = X_test1[:, :seed_timesteps, :]


# create a random search space
search_space = {
    'reservoir_size': [100, 200, 300, 400, 500, 600, 1000],
    'spectral_radius': [0.5, 0.9, 1.0, 1.1, 1.5],
    'sparsity': [0.05, 0.1, 0.15, 0.2],
    'leaking_rate': [0.1, 0.3, 0.5, 0.7],
    'connectivity': [0.05, 0.1, 0.2],
    'ridge_alpha': [1e-8, 1e-6, 1e-4]
}
# number of random samples
n_samples = 10

# create a list to store the results
results = []

for i in range(n_samples):
    # sample the hyperparameters
    hyperparams = {
        'reservoir_size': np.random.choice(search_space['reservoir_size']),
        'spectral_radius': np.random.choice(search_space['spectral_radius']),
        'sparsity': np.random.choice(search_space['sparsity']),
        'leaking_rate': np.random.choice(search_space['leaking_rate']),
        'connectivity': np.random.choice(search_space['connectivity']),
        'ridge_alpha': np.random.choice(search_space['ridge_alpha']),
        'pred_len': 1,
    }

    # create the ESN
    esn = ESNReservoir(io_size=4, **hyperparams)

    # train the ESN
    esn.fit(X_train1, y_train1)

    # generate the output
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

    # plot the data in 2D
    fig = plt.figure()
    plt.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    plt.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    plt.plot(warming_inputs[0, :, 0].numpy(), warming_inputs[0, :, 1].numpy(), label='Warmup')
    # plot the point where the lines end
    plt.scatter(X_t[0, -1, 0].numpy(), X_t[0, -1, 1].numpy(), label='End True', color='red')
    plt.scatter(X_gen[-1, 0], X_gen[-1, 1], label='End Generated', color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    X_t = X_t.squeeze(0).numpy()

    # calculate the RMSE
    rmse = np.sqrt(np.mean((X_gen - X_t) ** 2))

    # calculate the R^2 score
    ss_res = np.sum((X_t - X_gen) ** 2)
    ss_tot = np.sum((X_t - np.mean(X_t)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # store the results
    results.append({
        'hyperparams': hyperparams,
        'rmse': rmse,
        'r2': r2
    })

# plot the results
rmse = [r['rmse'] for r in results]
r2 = [r['r2'] for r in results]

plt.scatter(rmse, r2)
plt.xlabel('RMSE')
plt.ylabel('R^2')
plt.show()

# find the best hyperparameters
best_result = min(results, key=lambda x: x['rmse'])
best_hyperparams = best_result['hyperparams']
print('Best hyperparameters:', best_hyperparams)
print('RMSE:', best_result['rmse'])
print('R^2:', best_result['r2'])

# Best result found during computation:
# Best hyperparameters: {'reservoir_size': 200, 'spectral_radius': 1.1, 'sparsity': 0.15, 'leaking_rate': 0.1, 'connectivity': 0.1, 'ridge_alpha': 1e-06, 'pred_len': 1}
# RMSE: 0.03342189314935667
# R^2: 0.9970140478425422