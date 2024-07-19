import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import torch
from Reservoirs.ESNRidge import ESNReservoir
from Utils.DataLoader import loadData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

out_len = 100
variables=['x','y','z']

results = []
n_iters=5
wrmp=100

(input_fit, target_fit), _ = loadData("lorenz")
io_size = input_fit.size(1)
# input_fit = input_fit[::10]
input_fit = input_fit[:100]
input_fit = input_fit.unsqueeze(0)

torch.manual_seed(0)
for i in range(n_iters):
    model = ESNReservoir(io_size, 1024, pred_len=1, spectral_radius=0.9, connectivity=0.1, leaking_rate=0.5).to(device)
    result = model.thermalize(input_fit)
    results.append(result)

# plot the result in 2 separate plots one for x and y
plt.figure(figsize=(15, 15))
for var, varname in enumerate(variables):
    plt.subplot(len(variables), 1, var+1)
    plt.title(varname)
    for result in results:
        plt.plot(result[:, var].cpu().detach().numpy())
    plt.grid()
plt.show()
