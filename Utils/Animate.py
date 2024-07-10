import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import glob

# Number of datasets
num_datasets = 3  # Update this number if you add more datasets

# Define a color palette
colors = ['blue', 'orange', 'purple']  # Add more colors if needed

# Load data from CSV files
datasets = []
times = []
for i in range(num_datasets):
    df = pd.read_csv(f'Data/3BP_{i}.csv')
    datasets.append(df[['x', 'y', 'vx', 'vy']].values)
    times.append(df['time'].values)

# Determine the maximum length of datasets
max_length = max(len(t) for t in times)

# Load constants (assuming they are the same for all datasets)
constants = pd.read_csv('Data/constants.csv')
m1 = constants['mass'][0]
m2 = constants['mass'][1]
x1 = constants['x'][0]
y1 = constants['y'][0]
x2 = constants['x'][1]
y2 = constants['y'][1]

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Restricted Three-Body Problem')

# Plot the static elements (Earth and Sun)
earth = Circle((x1, y1), 5, color='green', edgecolor='black', label='Earth')
sun = Circle((x2, y2), 7, color='red', edgecolor='black', label='Sun')
ax.add_patch(earth)
ax.add_patch(sun)

# Prepare lines and scatter plots for each dataset
lines = [ax.plot([], [], color=colors[i % len(colors)], label=f'Trajectory {i}')[0] for i in range(num_datasets)]
scatters = [ax.scatter([], [], color=colors[i % len(colors)], s=30) for i in range(num_datasets)]
# ax.legend()
frame_step = 10  # Increase this number to skip more frames and speed up the video

def init():
    for line, scat in zip(lines, scatters):
        line.set_data([], [])
        scat.set_offsets(np.array([[], []]).T)
    return lines + scatters

def animate(i):
    frame = i * frame_step
    for idx, (line, scat, data, t) in enumerate(zip(lines, scatters, datasets, times)):
        if i < len(t):
            line.set_data(data[:i, 0], data[:i, 1])
            scat.set_offsets(data[i, :2])
        else:
            line.set_data(data[:, 0], data[:, 1])
            scat.set_offsets(data[-1, :2])
    return lines + scatters

ani = animation.FuncAnimation(fig, animate, frames=(max_length // frame_step), init_func=init, blit=True)
ani.save('Media/three_body_problem.mp4', writer='ffmpeg', fps=30, dpi=300)
