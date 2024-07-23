from Utils.DataGenerator import generate_data
from Utils.DataGenerator import lorenz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], int(1e8))
y0 = [1.0, 0.5, 1.0]

# generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

sampling = int(1e3)
# sample the data every "sampling" points
t = t[::sampling]
data = data[::sampling]

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df['time'] = t

df.to_csv('Data/lorenz_3.csv', index=False)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('Media/lorenz3D.png')
plt.close()
