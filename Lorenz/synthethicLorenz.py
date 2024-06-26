import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.DataGenerator import generate_data
from Utils.DataGenerator import lorenz

# Define the parameters for data generation
t_span = (0, 50)
y0 = [1.0, 1.0, 1.0] # 0.9 1.0 1.0 for testing
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

sampling = 100
# sample the data every 100 points
t = t[::sampling]
data = data[::sampling]

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df['time'] = t
df.to_csv('Data/lorenz_data.csv', index=False)

# plot the data in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('Media/lorenz3D.png')
plt.close()

# Plotting the generated data
plt.plot(t, data)
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Lorenz System')
plt.savefig('Media/lorenzVariables_test.png')
plt.close()