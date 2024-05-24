import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from tqdm import tqdm


print("Loading the dataset...")
# Load the dataset
data = pd.read_csv('TBP_dataset.csv')  # Update with the correct path

n = 10

# Extract positions and accelerations
r1_x = data['r1_x'].values.reshape(-1, n)
r1_y = data['r1_y'].values.reshape(-1, n)
r1_z = data['r1_z'].values.reshape(-1, n)
r2_x = data['r2_x'].values.reshape(-1, n)
r2_y = data['r2_y'].values.reshape(-1, n)
r2_z = data['r2_z'].values.reshape(-1, n)
r3_x = data['r3_x'].values.reshape(-1, n)
r3_y = data['r3_y'].values.reshape(-1, n)
r3_z = data['r3_z'].values.reshape(-1, n)

a1_x = data['a1_x'].values.reshape(-1, n)
a1_y = data['a1_y'].values.reshape(-1, n)
a1_z = data['a1_z'].values.reshape(-1, n)
a2_x = data['a2_x'].values.reshape(-1, n)
a2_y = data['a2_y'].values.reshape(-1, n)
a2_z = data['a2_z'].values.reshape(-1, n)
a3_x = data['a3_x'].values.reshape(-1, n)
a3_y = data['a3_y'].values.reshape(-1, n)
a3_z = data['a3_z'].values.reshape(-1, n)

# Choose one condition to animate (e.g., the first condition)
condition_idx = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
max_range = np.array([r1_x.ptp(), r1_y.ptp(), r1_z.ptp()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:1:2j]
Yb = 0.5 * max_range * np.mgrid[-1:1:2j]
Zb = 0.5 * max_range * np.mgrid[-1:1:2j]
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

ax.set_box_aspect([1,1,1])

ax.set_xlim(min(r1_x[condition_idx].min(), r2_x[condition_idx].min(), r3_x[condition_idx].min()),
            max(r1_x[condition_idx].max(), r2_x[condition_idx].max(), r3_x[condition_idx].max()))
ax.set_ylim(min(r1_y[condition_idx].min(), r2_y[condition_idx].min(), r3_y[condition_idx].min()),
            max(r1_y[condition_idx].max(), r2_y[condition_idx].max(), r3_y[condition_idx].max()))
ax.set_zlim(min(r1_z[condition_idx].min(), r2_z[condition_idx].min(), r3_z[condition_idx].min()),
            max(r1_z[condition_idx].max(), r2_z[condition_idx].max(), r3_z[condition_idx].max()))

line1, = ax.plot([], [], [], 'r-', label='Body 1')
line2, = ax.plot([], [], [], 'g-', label='Body 2')
line3, = ax.plot([], [], [], 'b-', label='Body 3')
quiver1 = ax.quiver([], [], [], [], [], [], color='r', length=0.1, normalize=True)
quiver2 = ax.quiver([], [], [], [], [], [], color='g', length=0.1, normalize=True)
quiver3 = ax.quiver([], [], [], [], [], [], color='b', length=0.1, normalize=True)
ax.legend()

# Function to create a sphere
def create_sphere(center, radius=0.1, color = 'r'):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return x, y, z

sphere1 = [ax.plot_surface(*create_sphere((0, 0, 0), color='r'), color='r', alpha=0.6)]
sphere2 = [ax.plot_surface(*create_sphere((0, 0, 0), color='g'), color='g', alpha=0.6)]
sphere3 = [ax.plot_surface(*create_sphere((0, 0, 0), color='b'), color='b', alpha=0.6)]

def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    line3.set_data([], [])
    line3.set_3d_properties([])
    if quiver1 in ax.collections:
        quiver1.remove()
    if quiver2 in ax.collections:
        quiver2.remove()
    if quiver3 in ax.collections:
        quiver3.remove()
    for s in sphere1 + sphere2 + sphere3:
        s.remove()
    return line1, line2, line3

print("Animating...")

pbar = tqdm(total=n, desc='Animating', unit='frame')

def update(frame):
    line1.set_data(r1_x[condition_idx, max(0, frame-10):frame + 1], r1_y[condition_idx, max(0, frame-10):frame + 1])
    line1.set_3d_properties(r1_z[condition_idx, max(0, frame-10):frame + 1])
    line2.set_data(r2_x[condition_idx, max(0, frame-10):frame + 1], r2_y[condition_idx, max(0, frame-10):frame + 1])
    line2.set_3d_properties(r2_z[condition_idx, max(0, frame-10):frame + 1])
    line3.set_data(r3_x[condition_idx, max(0, frame-10):frame + 1], r3_y[condition_idx, max(0, frame-10):frame + 1])
    line3.set_3d_properties(r3_z[condition_idx, max(0, frame-10):frame + 1])

    for s in sphere1 + sphere2 + sphere3:
        if s in ax.collections:
            s.remove()
    sphere1[0] = ax.plot_surface(*create_sphere((r1_x[condition_idx, frame], r1_y[condition_idx, frame], r1_z[condition_idx, frame]), color='r'), color='r', alpha=0.6)
    sphere2[0] = ax.plot_surface(*create_sphere((r2_x[condition_idx, frame], r2_y[condition_idx, frame], r2_z[condition_idx, frame]), color='g'), color='g', alpha=0.6)
    sphere3[0] = ax.plot_surface(*create_sphere((r3_x[condition_idx, frame], r3_y[condition_idx, frame], r3_z[condition_idx, frame]), color='b'), color='b', alpha=0.6)

    quiver1 = ax.quiver(r1_x[condition_idx, frame], r1_y[condition_idx, frame], r1_z[condition_idx, frame],
                        a1_x[condition_idx, frame], a1_y[condition_idx, frame], a1_z[condition_idx, frame], color='r',
                        length=0.1, normalize=True)

    quiver2 = ax.quiver(r2_x[condition_idx, frame], r2_y[condition_idx, frame], r2_z[condition_idx, frame],
                        a2_x[condition_idx, frame], a2_y[condition_idx, frame], a2_z[condition_idx, frame], color='g',
                        length=0.1, normalize=True)

    quiver3 = ax.quiver(r3_x[condition_idx, frame], r3_y[condition_idx, frame], r3_z[condition_idx, frame],
                        a3_x[condition_idx, frame], a3_y[condition_idx, frame], a3_z[condition_idx, frame], color='b',
                        length=0.1, normalize=True)

    pbar.update(1)

    return line1, line2, line3, quiver1, quiver2, quiver3

ani = animation.FuncAnimation(fig, update, frames=range(n), init_func=init, blit=False, repeat=False)

# Save the animation
ani.save('three_body_problem_3d_with_spheres_and_acceleration.mp4', writer='ffmpeg', fps=30, dpi = 300)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('3D Animation of the Three-Body Problem with Spheres and Acceleration')
#plt.show()
