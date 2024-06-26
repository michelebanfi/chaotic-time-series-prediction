from Utils.DataGenerator import generate_data
from Utils.DataGenerator import restricted_three_body
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for data generation
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 100000)
y0 = [60, 0, 5, 1.5]

# read constants
constants = pd.read_csv('Data/constants.csv')
m1 = constants['mass'][0]
m2 = constants['mass'][1]
x1 = constants['x'][0]
y1 = constants['y'][0]
x2 = constants['x'][1]
y2 = constants['y'][1]

# Generate the data
t, data = generate_data(restricted_three_body, t_span, y0, t_eval, args=(m1, m2, x1, y1, x2, y2))

sampling = 100

# sample the data every 100 points
t = t[::sampling]
data = data[::sampling]

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy'])
df['time'] = t
df.to_csv('Data/3BP.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Trajectory of the third body')
plt.scatter(data[-1, 0], data[-1, 1], color='blue', label='Cristiano Ronaldo', s=50)
plt.scatter([x1], [y1], color='green', label='Earth', s=100)
plt.scatter([x2], [y2], color='red', label='Sun', s=200)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Restricted Three-Body Problem')
plt.grid()
plt.savefig('Media/3BP.png')
plt.close()


# plot only the orbit
plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Trajectory of the third body')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Restricted Three-Body Problem')
plt.grid()
plt.savefig('Media/3BP_orbit.png')
plt.close()
