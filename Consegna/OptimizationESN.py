import random
from Reservoirs.ESNRidge import ESNReservoir
from Utils.DataLoader import loadData
import numpy as np
import torch
import matplotlib.pyplot as plt
from Utils.Losses import NormalizedMeanSquaredError as NMSE
import itertools
import os

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

problem = "MackeyGlass"
(input_fit, target_fit), (input_gen, target_gen), _ = loadData(problem, version=6)
io_size = input_fit.size(1)
input_fit = input_fit.unsqueeze(0)
target_fit = target_fit.unsqueeze(0)
input_gen = input_gen.unsqueeze(0)

# number of random samples if we are doing a random search
n_samples = 30

# search_space = {
#     'reservoir_size': [512, 1024, 1500, 2048],
#     'spectral_radius': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
#     'leaking_rate': [0.4, 0.5, 0.6, 0.7, 0.9],
#     'connectivity': [0.1, 0.15, 0.2],
#     'ridge_alpha': [1e-4, 1e-6, 1e-7, 1e-8]
# }

search_space = {
    'reservoir_size': [2048],
    'spectral_radius': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    'leaking_rate': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
    'connectivity': [0.2],
    'ridge_alpha': [1e-4, 1e-6, 1e-7, 1e-8]
}

# create a list to store the results
results = []


keys, values = zip(*search_space.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(permutations_dicts.__len__())
for i in range(permutations_dicts.__len__()):
    # sample the hyperparameters

    hyperparams = {
        'reservoir_size': permutations_dicts[i]['reservoir_size'],
        'spectral_radius': permutations_dicts[i]['spectral_radius'],
        'leaking_rate': permutations_dicts[i]['leaking_rate'],
        'connectivity': permutations_dicts[i]['connectivity'],
        'ridge_alpha': permutations_dicts[i]['ridge_alpha'],
        'pred_len': 1,
    }

    seed_torch()

    # create the ESN
    esn = ESNReservoir(io_size=io_size, **hyperparams)

    outputs = esn.fit(input_fit, target_fit)
    outputs = torch.tensor(outputs, dtype=torch.float32)

    nmse = NMSE(outputs, target_fit).item()
    print("NMSE: ", nmse)

    _, h = esn(input_gen)

    X_gen = np.zeros((target_gen.size(0), io_size))
    y = input_gen[:, -1, :].cpu().numpy()
    for i in range(target_gen.size(0)):
        input = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        output, h = esn(input, h)
        y = output[:, 0, :].cpu().numpy()
        z = output[0, 0, :].cpu().numpy()
        X_gen[i] = z


    if problem == "R3BP":

        # calculate the position of the earth and the
        # plot the data in 2D
        plt.figure(figsize=(10, 10))
        plt.plot(target_gen[:, 0], target_gen[:, 1], label='True')
        plt.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
        # plot the hyperparams in the photo
        plt.title(f"Reservoir Size: {hyperparams['reservoir_size']}, Spectral Radius: {hyperparams['spectral_radius']}, Leaking Rate: {hyperparams['leaking_rate']}, Connectivity: {hyperparams['connectivity']}, Ridge Alpha: {hyperparams['ridge_alpha']}")
        # draw the sun in these coordinates: (-2.217567596485422 -0.020616745263895776)
        plt.scatter(-2.217567596485422, -0.020616745263895776, label='Sun', color='red')
        # draw the earth in these coordinates: 1.0022193178526422 -0.020616745263895776
        plt.scatter(1.0022193178526422, -0.020616745263895776, label='Earth', color='green')
        plt.text(1, 1, nmse, bbox=dict(fill=False, edgecolor='red', linewidth=2))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid()
        plt.show()

    elif problem == "lorenz":
        # plot the data in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(target_gen[:, 0], target_gen[:, 1], target_gen[:, 2], label='True')
        ax.plot(X_gen[:, 0], X_gen[:, 1], X_gen[:, 2], label='Generated')
        plt.title(
            f"Reservoir Size: {hyperparams['reservoir_size']}, Spectral Radius: {hyperparams['spectral_radius']}, Leaking Rate: {hyperparams['leaking_rate']}, Connectivity: {hyperparams['connectivity']}, Ridge Alpha: {hyperparams['ridge_alpha']}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()
        plt.close()

    elif problem == "MackeyGlass":
        print("iteration done: ", i, " out of ", permutations_dicts.__len__(), "NMSE: ", nmse)
        # plot the data in 2D
        # plt.figure(figsize=(10, 10))
        # plt.plot(target_gen[:, 0], label='True')
        # plt.plot(X_gen[:, 0], label='Generated')
        # plt.title(
        #     f"Reservoir Size: {hyperparams['reservoir_size']}, Spectral Radius: {hyperparams['spectral_radius']}, Leaking Rate: {hyperparams['leaking_rate']}, Connectivity: {hyperparams['connectivity']}, Ridge Alpha: {hyperparams['ridge_alpha']}")
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # plt.grid()
        # plt.show()

    X_t = X_gen

    np_target_gen = target_gen.numpy()
    # calculate the RMSE
    rmse = np.sqrt(np.mean((X_gen - np_target_gen) ** 2))

    # calculate the R^2 score
    ss_res = np.sum((np_target_gen - X_gen) ** 2)
    ss_tot = np.sum((np_target_gen - np.mean(np_target_gen)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # store the results
    results.append({
        'hyperparams': hyperparams,
        'rmse': rmse,
        'r2': r2,
        'nmse': nmse
    })



# plot the results
rmse = [r['rmse'] for r in results]
r2 = [r['r2'] for r in results]

plt.scatter(rmse, r2)
plt.xlabel('RMSE')
plt.ylabel('R^2')
plt.show()

## choose wich values to display

# find the best hyperparameters
best_result = min(results, key=lambda x: x['nmse'])
best_hyperparams = best_result['hyperparams']
print('Best hyperparameters:', best_hyperparams)
print('NMSE:', best_result['nmse'])
# print('R^2:', best_result['r2'])

# best_result = max(results, key=lambda x: x['r2'])
# best_hyperparams = best_result['hyperparams']
# print('Best hyperparameters:', best_hyperparams)
# print('r2:', best_result['r2'])
#
# best_result = min(results, key=lambda x: x['rmse'])
# best_hyperparams = best_result['hyperparams']
# print('Best hyperparameters:', best_hyperparams)
# print('RMSE:', best_result['rmse'])
