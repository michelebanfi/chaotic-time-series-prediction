import sys
diego = True
if diego:
    sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
    from DataGenerator import generate_data
    from DataGenerator import restricted_three_body
else:
    from Utils.DataGenerator import generate_data
    from Utils.DataGenerator import restricted_three_body

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for data generation
t_span = (0, int(1e3))
t_eval = np.linspace(t_span[0], t_span[1], int(1e8))
y0 = [95, 0, 5, 5]

# read constants
if diego:
    constants = pd.read_csv('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Data/constants.csv')
else:
    constants = pd.read_csv("Data/constants.csv")
m1 = constants['mass'][0]
m2 = constants['mass'][1]
x1 = constants['x'][0]
y1 = constants['y'][0]
x2 = constants['x'][1]
y2 = constants['y'][1]

# Generate the data
t, data = generate_data(restricted_three_body, t_span, y0, t_eval, args=(m1, m2, x1, y1, x2, y2))

sampling = int(1e4)

# sample the data every ... points
t = t[::sampling]
data = data[::sampling]

# Save the data to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy'])
df['time'] = t
if diego:
    df.to_csv('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Data/3BP.csv', index=False)
else:
    df.to_csv("Data/3BP.csv", index = False)

plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Trajectory of the third body')
plt.scatter(data[-1, 0], data[-1, 1], color='blue', label='Third body', s=50)
plt.scatter([x1], [y1], color='green', label='Earth', s=100)
plt.scatter([x2], [y2], color='red', label='Sun', s=200)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Restricted Three-Body Problem')
plt.grid()
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Media/3BP.png')
else:
    plt.savefig("Media/3BP.png")
plt.close()


# plot only the orbit
plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Trajectory of the third body')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Restricted Three-Body Problem')
plt.grid()
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Media/3BP_orbit.png')
else:
    plt.savefig("Media/3BP_orbit.png")
plt.close()
