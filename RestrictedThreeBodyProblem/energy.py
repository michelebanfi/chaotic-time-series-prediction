import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv('Data/3BP.csv')
data = df[['x', 'y', 'vx', 'vy']].values
t = df['time'].values

# calculate the energy
constants = pd.read_csv('Data/constants.csv')
m1 = constants['mass'][0]
m2 = constants['mass'][1]
x1 = constants['x'][0]
y1 = constants['y'][0]
x2 = constants['x'][1]
y2 = constants['y'][1]

# calculate the potential energy we suppose the mass of the smaller body is 1
r1 = ((data[:, 0] - x1) ** 2 + (data[:, 1] - y1) ** 2) ** 0.5
r2 = ((data[:, 0] - x2) ** 2 + (data[:, 1] - y2) ** 2) ** 0.5
U = - m1 / r1 - m2 / r2

# calculate the kinetic energy, having that the velocity is calculated with the position and time
vx = data[:, 2]
vy = data[:, 3]
T = 0.5 * (vx ** 2 + vy ** 2)

# calculate the total energy
E = T + U

# plot the energy
plt.figure(figsize=(10, 6))
plt.plot(t, E, label='Total Energy')
plt.plot(t, T, label='Kinetic Energy')
plt.plot(t, U, label='Potential Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.title('Energy of the Restricted Three-Body Problem')
plt.grid()
plt.savefig('Media/3BP_energy.png')
plt.close()


