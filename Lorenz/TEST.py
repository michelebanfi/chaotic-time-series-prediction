import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
from Losses import NormalizedMeanSquaredError

sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Reservoirs")
from ESNTest import ESNTest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(f"D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Data/lorenz_0.csv")
data = torch.tensor(df[['x','y','z']].values).to(device).float()
data = data[::20]
# Define training setup
# criterion
n_input=2000
warmup=200
model=ESNTest(3, 2048, 3, component_perc=0.005, warmup=warmup, spectral_radius=0.99, sparsity=0).to(device)

n_init=100
input = data[n_init:n_init+n_input]
model.fit(input)

n_init=500
input = data[n_init:n_init+n_input]
target = input[warmup + 1:]

pred = model.predict(input)
criterion = NormalizedMeanSquaredError
loss = criterion(pred.unsqueeze(0), target.unsqueeze(0))
print("Loss:", loss.item())

for v in range(3):
    plt.subplot(1,3,v+1)
    plt.plot(range(n_input), input[:,v].cpu().detach(), label="True (with warmup)")
    plt.plot(range(1+warmup, n_input), pred[:,v].cpu().detach(), label="Predicted (no warmup)")
    plt.axvline(x=warmup, linestyle="--", color="gray")
plt.legend()
plt.show()
