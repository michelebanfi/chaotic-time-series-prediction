import torch
import pandas as pd
import matplotlib.pyplot as plt
from Reservoirs.LSTMReservoir import LSTMReservoir
from Reservoirs.ESNReservoir import ESNReservoir
from Reservoirs.GRUReservoir import GRUReservoir

# we want to demonstratre the EchoStateProperty for the LSTMReservoir

# take the data from the .csv file
df = pd.read_csv("../RestrictedThreeBodyProblem/Data/3BP_0.csv")
df = df[['x', 'y']]

results = []

for i in range(5):
    #local = df[i*100:i*100 + 100]
    local = df[:10]

    tn = torch.tensor(local.values).float().unsqueeze(0)
    lstm = ESNReservoir(2, 1024, 2, 10000)
    lstm.eval()
    # lstm = ESNReservoir(2, 8, 2, 1000)
    result = lstm(tn)
    results.append(result)

# plot the result in 2 separate plots one for x and y
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
#plt.ylim(-0.5, 0.5)
plt.title("x")
for result in results:
    plt.plot(result[0, :, 0].detach().numpy())
plt.grid()
plt.subplot(2, 1, 2)
#plt.ylim(-0.5, 0.5)
plt.title("y")
for result in results:
    plt.plot(result[0, :, 1].detach().numpy())
plt.grid()
plt.show()
