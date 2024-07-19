from Reservoirs.ESNRidge import ESNReservoir
from Utils.DataLoader import loadData
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

problem = "MackeyGlass"
(input_fit, target_fit), (input_gen, target_gen), scaler = loadData(problem, version=0)
io_size = input_fit.size(1)
n_input_gen = input_gen.size(0)
input_fit = input_fit.unsqueeze(0)
target_fit = target_fit.unsqueeze(0)
input_gen = input_gen.unsqueeze(0)
n_gen = target_gen.size(0)

if problem == "R3BP":
    # scale the position of earth and sun
    earthx = 90.909090
    earthy = 0

    sunx = -9.090909
    suny = 0

    # Updated coordinates array with placeholder values for the additional features
    coordinates = np.array([[earthx, earthy, 0, 0], [sunx, suny, 0, 0]])

    # Scale the coordinates using the fitted scaler
    scaled_coordinates = scaler.transform(coordinates)

    # Extract the scaled coordinates (ignoring the placeholder features)
    scaled_earthx, scaled_earthy = scaled_coordinates[0][:2]
    scaled_sunx, scaled_suny = scaled_coordinates[1][:2]

    print(scaled_earthx, scaled_earthy, scaled_sunx, scaled_suny)

reservoir_size = 64
pred_len = 1
spectral_radius = 0.3
leaking_rate = 1
connectivity = 0.1
ridge_alpha = 1e-8

esn = ESNReservoir(io_size, reservoir_size, pred_len, spectral_radius=spectral_radius,
                   leaking_rate=leaking_rate, connectivity=connectivity, ridge_alpha=ridge_alpha)

esn.fit(input_fit, target_fit)

# take the last target_gen.size(0) as input from input_gen
test_input = input_gen[:, -target_gen.size(0):, :]
test_target = target_gen

output, _ = esn(test_input)

# plot the result
plt.figure(figsize=(15, 10))
plt.plot(output[0, :, 0], label='generated')
plt.plot(target_gen[:, 0], label='true')
plt.legend()
plt.grid()
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Mackey-Glass System - Generated vs True')
plt.show()


# Generate data
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
    # plot the data in 2D
    plt.figure(figsize=(10, 10))
    plt.plot(target_gen[:, 0], target_gen[:, 1], label='True')
    plt.plot(X_gen[:, 0], X_gen[:, 1], label='Generated')
    plt.scatter(scaled_sunx, scaled_suny, label='Sun', color='red')
    plt.scatter(scaled_earthx, scaled_earthy, label='Earth', color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

elif problem == "lorenz":
    # plot the data in 3D
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(target_gen[:, 0], target_gen[:, 1], target_gen[:, 2], label='True')
    ax.plot(X_gen[:, 0], X_gen[:, 1], X_gen[:, 2], label='Generated')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig("../Media/Generated_Lorenz.png")
    plt.close()

    # plot also the three subplots plots still with the input gen
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    axs[0].plot(range(n_input_gen), input_gen[0, :, 0], label='Input')
    axs[0].plot(range(n_input_gen, n_input_gen + n_gen), target_gen[:, 0], label="True", linestyle="--")
    axs[0].plot(range(n_input_gen, n_input_gen + n_gen), X_gen[:, 0], label='Generated')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(range(n_input_gen), input_gen[0, :, 1], label='Input')
    axs[1].plot(range(n_input_gen, n_input_gen + n_gen), target_gen[:, 1], label="True", linestyle="--")
    axs[1].plot(range(n_input_gen, n_input_gen + n_gen), X_gen[:, 1], label='Generated')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Z')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(range(n_input_gen), input_gen[0, :, 2], label='Input')
    axs[2].plot(range(n_input_gen, n_input_gen + n_gen), target_gen[:, 2], label="True", linestyle="--")
    axs[2].plot(range(n_input_gen, n_input_gen + n_gen), X_gen[:, 2], label='Generated')
    axs[2].set_xlabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].legend()
    axs[2].grid()

    plt.savefig("../Media/Generated_Lorenz_subplots.png")
    plt.close()

elif problem == "MackeyGlass":

    # plot the data
    plt.figure(figsize=(15, 10))
    plt.plot(range(n_input_gen), input_gen[0, :, 0], label='Input')
    plt.plot(range(n_input_gen, n_input_gen + n_gen), target_gen[:, 0], label="True", linestyle="--")
    plt.plot(range(n_input_gen, n_input_gen + n_gen), X_gen[:, 0], label='Generated')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.show()

    # # plot the 2D phase plot
    # tau = 17
    # plt.figure(figsize=(20, 20))
    # plt.plot(X_gen[:-tau, 0], X_gen[tau:, 0])
    # plt.xlabel('x(t)')
    # plt.ylabel('x(t-τ)')
    # plt.title('Mackey-Glass System - 2D Phase Plot (100,000 points, generated)')
    # plt.grid()
    # plt.show()



