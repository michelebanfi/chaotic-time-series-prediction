import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import torch
from Reservoirs.ESNPCA import ESNPCA
from Utils.DataLoader import loadData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

out_len = 100
variables=['x','y','z']
io_size = len(variables)

results = []
n_iters=5
wrmp=100

(input_fit, target_fit), _ = loadData("lorenz")

torch.manual_seed(0)
for i in range(n_iters):
    model = ESNPCA(io_size, io_size, 2048, warmup=wrmp, spectral_radius=0.9, sparsity=0.3, leaking_rate=0.5).to(device)
    result = model.thermalize(input_fit)
    results.append(result)

# plot the result in 2 separate plots one for x and y
plt.figure(figsize=(15, 15))
for var, varname in enumerate(variables):
    plt.subplot(io_size, 1, var+1)
    plt.title(varname)
    for result in results:
        plt.plot(result[:, var].cpu().detach().numpy())
    plt.grid()
plt.show()
