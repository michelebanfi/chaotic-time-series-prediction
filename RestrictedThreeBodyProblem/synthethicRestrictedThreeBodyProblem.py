from Utils.DataGenerator import generate_data
from Utils.DataGenerator import restricted_three_body
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for data generation
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 100000)
y0 = [1.2, 0, 0, 1]


# Generate the data
t, data = generate_data(restricted_three_body, t_span, y0, t_eval)

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'ax', 'ay'])
df['time'] = t
df.to_csv('Data/3BP.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Trajectory of the third body')
plt.scatter([1, -1], [1, -1], color='red', label='Primary bodies')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Restricted Three-Body Problem')
plt.grid()
plt.show()
plt.savefig('Media/3BP.png')
plt.close()
