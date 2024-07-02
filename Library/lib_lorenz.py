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
    leak_rate = 0.3
    spectral_radius = 1.25
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 1e-8

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


if __name__ == '__main__':
    main()