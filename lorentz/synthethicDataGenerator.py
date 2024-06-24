import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_data(equation, t_span, y0, t_eval):
    sol = solve_ivp(equation, t_span, y0, t_eval=t_eval)
    return sol.t, sol.y.T

# Define the parameters for data generation
t_span = (0, 50)
y0 = [1.0, 1.0, 1.0]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df['time'] = t
df.to_csv('lorenz_data.csv', index=False)

# plot the data in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('lorenz3D.png')
plt.close()


# Plotting the generated data
plt.plot(t, data)
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Lorenz System')
plt.savefig('lorenzVariables.png')
plt.close()


# import matplotlib.pyplot as plt
# import numpy as np
#
# def lorenz(xyz, *, s=10, r=28, b=2.667):
#     """
#     Parameters
#     ----------
#     xyz : array-like, shape (3,)
#        Point of interest in three-dimensional space.
#     s, r, b : float
#        Parameters defining the Lorenz attractor.
#
#     Returns
#     -------
#     xyz_dot : array, shape (3,)
#        Values of the Lorenz attractor's partial derivatives at *xyz*.
#     """
#     x, y, z = xyz
#     x_dot = s*(y - x)
#     y_dot = r*x - y - x*z
#     z_dot = x*y - b*z
#     return np.array([x_dot, y_dot, z_dot])
#
# dt = 0.01
# num_steps = 10000
#
# xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
# xyzs[0] = (0., 1., 1.05)  # Set initial values
# # Step through "time", calculating the partial derivatives at the current point
# # and using them to estimate the next point
# for i in range(num_steps):
#     xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
#
# # Plot
# ax = plt.figure().add_subplot(projection='3d')
#
# ax.plot(*xyzs.T, lw=0.6)
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Lorenz Attractor")
#
# plt.show()