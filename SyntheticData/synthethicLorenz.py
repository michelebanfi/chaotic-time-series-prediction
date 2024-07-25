from Utils.DataGenerator import generate_data
from Utils.DataGenerator import lorenz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


results = []

t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], int(1e4))
y0 = [1.0, 1.0, 1.0]

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

# sampling = int(1e3)
# # sample the data every ... points
# t = t[::sampling]
# data = data[::sampling]

# Save the data to CSV
# df = pd.DataFrame(data, columns=['x', 'y', 'z'])
# df['time'] = t

# df.to_csv('Data/lorenz_3.csv', index=False)

results.append(data)
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], int(1e4))
y0 = [1.0, 1.0000001, 1.0]

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

results.append(data)

t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], int(1e4))
y0 = [1.0, 1.0000000001, 1.0]

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

results.append(data)

# plot the data in a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(data[:, 0], data[:, 1], data[:, 2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# #plt.savefig('Media/lorenz3D.png')
# plt.show()

variables = ['x', 'y', 'z']
initial_conditions = [[1.0, 1.0, 1.0], [1.0, 1.0000001, 1.0], [1.0, 1.0000000001, 1.0]]

# # Create subplots, one for each variable of the Lorenz System, 3 in total. Then for each sublplot, iterate
# # over the results and plot the data
# fig, axs = plt.subplots(3, 1, figsize=(20, 20))
# for i in range(3):
#     axs[i].plot(t, results[:, 0])
# plt.show()

# create one figure and plot the three generated series, but plot only the first variable
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.plot(t, results[i][:, 0], label=f"Initial Condition: {initial_conditions[i]}")
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.grid()
plt.show()