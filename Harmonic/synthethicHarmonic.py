import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.DataGenerator import generate_data
from Utils.DataGenerator import harmonic_oscillator

# Define the parameters for data generation
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

y0 = [1.0, 0.0]

# Generate the data for the harmonic oscillator
t, data = generate_data(harmonic_oscillator, t_span, y0, t_eval)

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'v'])
df['time'] = t
df.to_csv('harmonic_data.csv', index=False)

# plot the data in a 2D plot
plt.plot(t, data)
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Harmonic Oscillator')
plt.savefig('harmonicVariables.png')
plt.close()