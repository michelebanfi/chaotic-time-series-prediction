import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import ESN
from reservoirpy.datasets import to_forecasting
import reservoirpy as rpy
import pandas as pd

def main():
    data = pd.read_csv("3BP_0.csv")
    variables = ['x', 'y', 'vx', 'vy']

    X = data[variables].values

    # scale the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    n = int(len(X) * 0.8)

    # plot the 3 body problem in 2d
    plt.plot(X[:n, 0], X[:n, 1])
    plt.plot(X[n:, 0], X[n:, 1])
    plt.scatter(X[-1, 0], X[-1, 1], c='r')
    plt.legend(['Train', 'Test', 'Last point'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.close()

    x, y = to_forecasting(X, forecast=1)
    X_train1, y_train1 = x[:n], y[:n]
    X_test1, y_test1 = x[n:], y[n:]

    units = 500
    leak_rate = 0.3
    spectral_radius = 1.25
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 1e-8

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity)

    readout = Ridge(4, ridge=regularization)

    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

    # train the ESN
    esn.fit(X_train1, y_train1)

    # predict the test data
    predicted = esn.run(X_test1)

    # compute the NRMSE
    nrmse_value = nrmse(y_test1, predicted)
    r2 = rsquare(y_test1, predicted)

    print(f"NRMSE: {nrmse_value:.4f}")
    print(f"R^2: {r2:.4f}")

    # Plot the predicted trajectory
    plt.plot(predicted[:, 0], predicted[:, 1], 'r')
    plt.plot(y_test1[:, 0], y_test1[:, 1], 'b')
    # plot a point in the last position
    plt.scatter(y_test1[-1, 0], y_test1[-1, 1], c='g')
    plt.scatter(predicted[-1, 0], predicted[-1, 1], c='r')
    plt.legend(['Predicted', 'True', 'Last True point', 'Last Predicted point'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def generation():
    data = pd.read_csv("3BP_0.csv")
    variables = ['x', 'y', 'vx', 'vy']

    X = data[variables].values

    # scale the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    n = int(len(X) * 0.8)

    # plot the 3 body problem in 2d
    plt.plot(X[:n, 0], X[:n, 1])
    plt.plot(X[n:, 0], X[n:, 1])
    plt.scatter(X[-1, 0], X[-1, 1], c='r')
    plt.legend(['Train', 'Test', 'Last point'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.close()

    x, y = to_forecasting(X, forecast=1)
    X_train1, y_train1 = x[:n], y[:n]
    X_test1, y_test1 = x[n:], y[n:]

    units = 500
    leak_rate = 0.3
    spectral_radius = 1.25
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 1e-8

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity)

    readout = Ridge(4, ridge=regularization)

    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

    # train the ESN
    esn.fit(X_train1, y_train1)

    seed_timesteps = 100

    warming_inputs = X_test1[:seed_timesteps]

    warming_out = esn.run(warming_inputs, reset=True)

    nb_generations = 100

    X_gen = np.zeros((nb_generations, 4))
    y = warming_out[-1]
    for t in range(nb_generations):  # generation
        y = esn(y)
        X_gen[t, :] = y

    X_t = X_test1[seed_timesteps: nb_generations + seed_timesteps]

    # Plot the predicted trajectory
    plt.plot(X_gen[:, 0], X_gen[:, 1], 'r')
    plt.plot(X_t[:, 0], X_t[:, 1], 'b')
    # plot a point in the last position
    plt.scatter(X_t[-1, 0], X_t[-1, 1], c='g')
    plt.scatter(X_gen[-1, 0], X_gen[-1, 1], c='r')
    plt.legend(['Predicted', 'True', 'Last True point', 'Last Predicted point'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    main()