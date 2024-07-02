import sys

diego = False
if diego:
    sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
    from DataGenerator import generate_data
    from DataGenerator import lorenz
else:
    from Utils.DataGenerator import generate_data
    from Utils.DataGenerator import lorenz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for data generation
t_span = (0, int(1e2))
t_eval = np.linspace(t_span[0], t_span[1], int(1e8))
y0 = [1, 0.9, 1]

# Generate the data
t, data = generate_data(lorenz, t_span, y0, t_eval)

sampling = int(1e5)
# sample the data every ... points
t = t[::sampling]
data = data[::sampling]

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df['time'] = t
if diego:
    df.to_csv('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Data/lorenz_data.csv', index=False)
else:
    df.to_csv('Data/lorenz_2.csv', index=False)

# plot the data in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Media/lorenz3D.png')
else:
    plt.savefig('Media/lorenz3D.png')
plt.close()

# Plotting the generated data
plt.plot(t, data)
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Lorenz System')
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Media/lorenzVariables_test.png')
else:
    plt.savefig("Media/lorenz3D.png")
plt.close()