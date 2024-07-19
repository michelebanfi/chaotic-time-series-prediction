import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import torch
from Reservoirs.ESNPCA import ESNPCA
from Utils.DataLoader import loadData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

variables=['Component1','Component2','Component3']

results = []
n_iters=5
wrmp=100

(input_fit, target_fit), _ = loadData("lorenz")
io_size = input_fit.size(1)

for i in range(n_iters):
    model = ESNPCA(io_size, io_size, 512, warmup=wrmp, spectral_radius=0.9, sparsity=0.1, leaking_rate=1, seed=None).to(device)
    result = model.thermalize(input_fit)
    results.append(result)

plt.figure(figsize=(15, 15))
for var, varname in enumerate(variables):
    plt.subplot(len(variables), 1, var+1)
    plt.title(varname)
    for result in results:
        # result = result[:50, :]
        plt.plot(result[:, var].cpu().detach().numpy())
    plt.grid()
plt.show()
