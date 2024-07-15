import sys
import os
sys.path.append(os.getcwd())

from Hyperoptimization.hyperoptESNPCA import optimize
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Reservoirs.ESNPCA import ESNPCA
import matplotlib.pyplot as plt

## WHOLE DATA
data_filename = "3BP_0"
df = pd.read_csv(f"Data/R3BP/{data_filename}.csv")
data = df[['x','y']].values
data = data[::10]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = torch.tensor(data).float()
io_size = data.size(1)

# FIT
n_init_fit=int(1e3)
n_input_fit=int(5e4)
input_fit = data[n_init_fit:n_init_fit+n_input_fit]
target_fit = data[n_init_fit+1:n_init_fit+n_input_fit+1]

## GENERATION
n_init_gen=int(6e4)
n_input_gen=int(5e3)
n_gen=int(1e4)
input_gen = data[n_init_gen:n_init_gen+n_input_gen]
target_gen = data[n_init_gen+n_input_gen:n_init_gen+n_input_gen+n_gen]

search_space = {
    'components':[5, 10, 15, 20], 
    'spectral_radius':[0.2*x for x in range(3,7)], 
    'sparsity':[0.3,0.5,0.7],  
    'leaking_rate':[0.2*x for x in range(1,5)],
    'memory': [500]
}

model, args, loss, pred_fit, pred_gen = optimize(model_class=ESNPCA, 
                             input_size=io_size, reservoir_size=1024, output_size=io_size, 
                             data_train=(input_fit, target_fit), data_generation=(input_gen, target_gen), 
                             nextractions=100, ntests=3,
                             model_savepath= "Models/R3BP/ESNPCA_"+data_filename+"_best_model.pth", 
                             **search_space)
# ## NOT BEST Loss: 0.9961 with {'components': 10, 'spectral_radius': 0.8, 'sparsity': 0.3, 'leaking_rate': 0.2} - reservoir_size=1024 - 3BP_0
print(f"Loss: {loss:.4f}", "-", f"Parameters: {args}")
torch.save(model.state_dict(), "Models/R3BP/ESNPCA_"+data_filename+"_best_model.pth")

plt.figure(figsize=(15,15))
plt.title("Training")
for v in range(io_size):
    plt.subplot(io_size,1,v+1)
    plt.plot(target_fit[:,v].cpu(), label="Target", linestyle="--")
    plt.plot(pred_fit[:,v].cpu(), label="Predicted")
plt.legend()
plt.savefig("Media/R3BP/ESNPCA_fitting.png")
plt.close()

plt.figure(figsize=(15,15))
plt.title("Generation")
for v in range(io_size):
    plt.subplot(io_size,1,v+1)
    plt.plot(range(n_input_gen), input_gen[:,v].cpu(), label="Input")
    plt.plot(range(n_input_gen, n_input_gen+n_gen), target_gen[:,v].cpu(), label="Target", linestyle="--")
    plt.plot(range(n_input_gen, n_input_gen+n_gen), pred_gen[:,v].cpu(), label="Predicted")
plt.legend()
plt.savefig("Media/R3BP/ESNPCA_generation.png")
plt.close()