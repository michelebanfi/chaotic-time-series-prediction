import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('TBP_dataset.csv')  # Update with the correct path

n = 10000

# Extract positions and accelerations
r1_x = data['r1_x']
r1_y = data['r1_y']
r1_z = data['r1_z']
r2_x = data['r2_x']
r2_y = data['r2_y']
r2_z = data['r2_z']
r3_x = data['r3_x']
r3_y = data['r3_y']
r3_z = data['r3_z']

# extract velocities
v1_x = data['v1_x']
v1_y = data['v1_y']
v1_z = data['v1_z']
v2_x = data['v2_x']
v2_y = data['v2_y']
v2_z = data['v2_z']
v3_x = data['v3_x']
v3_y = data['v3_y']
v3_z = data['v3_z']

# Compute energies
m1 = 1
m2 = 1
m3 = 1

# Kinetic energy
T1 = 0.5 * m1 * (v1_x**2 + v1_y**2 + v1_z**2)
T2 = 0.5 * m2 * (v2_x**2 + v2_y**2 + v2_z**2)
T3 = 0.5 * m3 * (v3_x**2 + v3_y**2 + v3_z**2)
T = T1 + T2 + T3

# Potential energy
G = 1
r12 = np.sqrt((r1_x - r2_x)**2 + (r1_y - r2_y)**2 + (r1_z - r2_z)**2)
r13 = np.sqrt((r1_x - r3_x)**2 + (r1_y - r3_y)**2 + (r1_z - r3_z)**2)
r23 = np.sqrt((r2_x - r3_x)**2 + (r2_y - r3_y)**2 + (r2_z - r3_z)**2)
U = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)

# Total energy
E = T + U

print(E, E.shape)
plt.plot(E)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Total Energy')
plt.savefig('Energy.png')
plt.show()
