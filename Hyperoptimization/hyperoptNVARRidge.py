import pandas as pd
import numpy as np
import torch
from RidgeBasedExp.Modules.NVARRidge import NVARReservoir
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("../../RestrictedThreeBodyProblem/Data/3BP_0.csv")
X = df[['x', 'y', 'vx', 'vy']].values
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

# Define search space for NVAR
search_space = {
    'degree': [1, 2, 3],
    'delay': [1, 2],
    'ridge_alpha': [1e-8],
    'stride': [1, 2]
}

n_samples = 3
results = []

for i in range(n_samples):
    # Sample hyperparameters
    hyperparams = {
        'degree': np.random.choice(search_space['degree']),
        'delay': np.random.choice(search_space['delay']),
        'ridge_alpha': np.random.choice(search_space['ridge_alpha']),
        'stride': np.random.choice(search_space['stride'])
    }

    # Create and train NVAR model
    nvar = NVARReservoir(io_size=io_size, **hyperparams)
    nvar.fit(X_train1, y_train1)

    # Generate output
    X_gen = np.zeros((nb_generations, io_size))
    y = warming_inputs[:, -hyperparams['delay']:, :].squeeze(0).detach().numpy()
    for t in range(nb_generations):
        input = torch.tensor(y, dtype=torch.float32)
        if input.dim() == 2:
            input = input.unsqueeze(0)  # Add batch dimension if needed
        output = nvar(input)
        y = np.roll(y, -1, axis=0)
        y[-1] = output[0, -1, :].detach().numpy()
        X_gen[t] = y[-1]  # This should now work correctly

    X_t = X_test1[:, seed_timesteps:seed_timesteps + nb_generations * hyperparams['stride']:hyperparams['stride']]

    # Plot results
    fig = plt.figure()
    plt.plot(X_t[0, :, 0].numpy(), X_t[0, :, 1].numpy(), label='True')
    plt.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    plt.plot(warming_inputs[0, :, 0].numpy(), warming_inputs[0, :, 1].numpy(), label='Warmup')
    plt.scatter(X_t[0, -1, 0].numpy(), X_t[0, -1, 1].numpy(), label='End True', color='red')
    plt.scatter(X_gen[-1, 0], X_gen[-1, 1], label='End Generated', color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    X_t = X_t.squeeze(0).numpy()

    # Calculate metrics
    rmse = np.sqrt(np.mean((X_gen - X_t) ** 2))
    ss_res = np.sum((X_t - X_gen) ** 2)
    ss_tot = np.sum((X_t - np.mean(X_t)) ** 2)
    r2 = 1 - ss_res / ss_tot

    results.append({
        'hyperparams': hyperparams,
        'rmse': rmse,
        'r2': r2
    })

# Plot results
rmse = [r['rmse'] for r in results]
r2 = [r['r2'] for r in results]
plt.scatter(rmse, r2)
plt.xlabel('RMSE')
plt.ylabel('R^2')
plt.show()

# Find best hyperparameters
best_result = min(results, key=lambda x: x['rmse'])
best_hyperparams = best_result['hyperparams']
print('Best hyperparameters:', best_hyperparams)
print('RMSE:', best_result['rmse'])
print('R^2:', best_result['r2'])