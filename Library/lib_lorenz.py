import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import ESN
from reservoirpy.datasets import to_forecasting
import reservoirpy as rpy

def main():
    timesteps = 2000
    X = lorenz(timesteps)

    # scale the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Plot the Lorenz attractor in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()

    n = int(timesteps * 0.8)

    x, y = to_forecasting(X, forecast=1)
    X_train1, y_train1 = x[:n], y[:n]
    X_test1, y_test1 = x[n:], y[n:]

    units = 500
    leak_rate = 0.4865716163619631
    spectral_radius = 1.367163082884632
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 0.06277358548329898

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity)

    readout = Ridge(3, ridge=regularization)

    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

    # train the ESN
    esn.fit(X_train1, y_train1)

    # predict the test data
    predicted = esn.run(X_test1)

    # compute the NRMSE
    nrmse_value = nrmse(X_test1, predicted)
    r2 = rsquare(X_test1, predicted)

    print(f"NRMSE: {nrmse_value:.4f}")
    print(f"R^2: {r2:.4f}")

    # Plot the Lorenz attractor in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], label="Predicted")
    ax.plot(X_test1[:, 0], X_test1[:, 1], X_test1[:, 2], label="True")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def generation():
    timesteps = 5000
    X = lorenz(timesteps)

    # scale the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    n = int(len(X) * 0.8)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    plt.close()

    x, y = to_forecasting(X, forecast=1)
    X_train1, y_train1 = x[:n], y[:n]
    X_test1, y_test1 = x[n:], y[n:]

    units = 500
    leak_rate = 0.4865716163619631
    spectral_radius = 1.367163082884632
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 0.06277358548329898

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity)

    readout = Ridge(3, ridge=regularization)

    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

    # train the ESN
    esn.fit(X_train1, y_train1)

    seed_timesteps = 100

    warming_inputs = X_test1[:seed_timesteps]

    warming_out = esn.run(warming_inputs, reset=True)

    nb_generations = 100

    X_gen = np.zeros((nb_generations, 3))
    y = warming_out[-1]
    for t in range(nb_generations):  # generation
        y = esn(y)
        X_gen[t, :] = y

    X_t = X_test1[seed_timesteps: nb_generations + seed_timesteps]

    # plot the 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_gen[:, 0], X_gen[:, 1], X_gen[:, 2], 'r')
    ax.plot(X_t[:, 0], X_t[:, 1], X_t[:, 2], 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(['Generated', 'True'])
    plt.show()



if __name__ == '__main__':
    generation()