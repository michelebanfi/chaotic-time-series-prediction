from Reservoirs.ESNRidge import ESNReservoir
from Utils.DataLoader import loadData
import numpy as np
import torch
import matplotlib.pyplot as plt

problem = "lorenz"
(input_fit, target_fit), (input_gen, target_gen) = loadData(problem)
io_size = input_fit.size(1)
input_fit = input_fit.unsqueeze(0)
target_fit = target_fit.unsqueeze(0)
input_gen = input_gen.unsqueeze(0)

reservoir_size = 400
pred_len = 1
spectral_radius = 1.5
leaking_rate = 0.5
connectivity = 0.2
ridge_alpha = 1e-04

esn = ESNReservoir(io_size, reservoir_size, pred_len, spectral_radius=spectral_radius,
                   leaking_rate=leaking_rate, connectivity=connectivity, ridge_alpha=ridge_alpha)

esn.fit(input_fit, target_fit)

# Generate data
_, h = esn(input_gen)
X_gen = np.zeros((input_gen.size(1), io_size))
y = input_gen[:, -1, :].cpu().numpy()
for i in range(input_gen.size(1)):
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

