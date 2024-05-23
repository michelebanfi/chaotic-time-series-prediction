import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the dataset
data = pd.read_csv('TBP_dataset.csv')  # Update with the correct path

# Extract positions
r1_x = data['r1_x'].values.reshape(-1, 1000000)
r1_y = data['r1_y'].values.reshape(-1, 1000000)
r2_x = data['r2_x'].values.reshape(-1, 1000000)
r2_y = data['r2_y'].values.reshape(-1, 1000000)
r3_x = data['r3_x'].values.reshape(-1, 1000000)
r3_y = data['r3_y'].values.reshape(-1, 1000000)

# Choose one condition to animate (e.g., the first condition)
condition_idx = 0

fig, ax = plt.subplots(figsize=(20, 20))
ax.set_xlim(min(r1_x[condition_idx].min(), r2_x[condition_idx].min(), r3_x[condition_idx].min()),
            max(r1_x[condition_idx].max(), r2_x[condition_idx].max(), r3_x[condition_idx].max()))
ax.set_ylim(min(r1_y[condition_idx].min(), r2_y[condition_idx].min(), r3_y[condition_idx].min()),
            max(r1_y[condition_idx].max(), r2_y[condition_idx].max(), r3_y[condition_idx].max()))

line1, = ax.plot([], [], 'r-', label='Body 1')  # Changed marker to line
line2, = ax.plot([], [], 'g-', label='Body 2')  # Changed marker to line
line3, = ax.plot([], [], 'b-', label='Body 3')  # Changed marker to line
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def update(frame):
    line1.set_data(r1_x[condition_idx, :frame+1], r1_y[condition_idx, :frame+1])
    line2.set_data(r2_x[condition_idx, :frame+1], r2_y[condition_idx, :frame+1])
    line3.set_data(r3_x[condition_idx, :frame+1], r3_y[condition_idx, :frame+1])
    return [line1, line2, line3]

ani = animation.FuncAnimation(fig, update, frames=range(1000000), init_func=init, blit=True, repeat=False)

# Save the animation
ani.save('three_body_problem.mp4', writer='ffmpeg', fps=30)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('2D Animation of the Three-Body Problem')
