import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jitcdde import jitcdde, y, t

# Define the Mackey-Glass equation
def mackey_glass():
    beta, gamma, tau = 0.2, 0.1, 17
    yield beta * y(0, t - tau) / (1 + y(0, t - tau)**10) - gamma * y(0)

# Set up the DDE
dde = jitcdde(mackey_glass)

# Initial condition
dde.constant_past([1.12])

# Solve and sample
times = np.linspace(0, 1000, 10001)
solution = []

for time in times:
    solution.append(dde.integrate(time)[0])

# Convert to numpy array
solution = np.array(solution)

# Sample the data
sampling = 1
t_sampled = times[::sampling]
y_sampled = solution[::sampling]

# Save the data to CSV
df = pd.DataFrame({'t': t_sampled, 'x': y_sampled})
df.to_csv('../Data/MackeyGlass/mackey_glass_1.csv', index=False)

# Plot the data
plt.figure(figsize=(15, 10))
plt.plot(t_sampled, y_sampled)
plt.xlabel('Time')
plt.ylabel('x')
plt.grid()
plt.title('Mackey-Glass System')
plt.show()

# Create 2D phase plot
tau = 17  # This is the delay used in the Mackey-Glass equation
plt.figure(figsize=(20, 20))
plt.plot(y_sampled[:-tau], y_sampled[tau:])
plt.xlabel('x(t)')
plt.ylabel('x(t-τ)')
plt.title('Mackey-Glass System - 2D Phase Plot (100,000 points, sampled)')
plt.grid()
plt.show()