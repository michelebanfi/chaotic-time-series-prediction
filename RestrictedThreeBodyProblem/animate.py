import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

# Load data from CSV
df = pd.read_csv('Data/3BP.csv')
data = df[['x', 'y', 'vx', 'vy']].values
t = df['time'].values

constants = pd.read_csv('Data/constants.csv')
m1 = constants['mass'][0]
m2 = constants['mass'][1]
x1 = constants['x'][0]
y1 = constants['y'][0]
x2 = constants['x'][1]
y2 = constants['y'][1]

# create an animation and save it as .mp4 file

fig, ax = plt.subplots()
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Restricted Three-Body Problem')

# plot the trajectory of the third body
line, = ax.plot([], [], label='Trajectory of the earth')
scat = ax.scatter([], [], color='blue', label='Moon', s=30)
earth = Circle((x1, y1), 5, color='green', edgecolor='black', label='Earth')  # Adjusted size of Earth
sun = Circle((x2, y2), 7, color='red', edgecolor='black', label='Sun')        # Adjusted size of Sun
ax.add_patch(earth)
ax.add_patch(sun)
ax.legend()

def init():
    line.set_data([], [])
    scat.set_offsets(np.array([[], []]).T)
    return line, scat

def animate(i):
    line.set_data(data[:i, 0], data[:i, 1])
    scat.set_offsets(data[i, :2])
    return line, scat

ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, blit=True)
ani.save('Media/three_body_problem.mp4', writer='ffmpeg', fps=30, dpi=300)

