import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
from Losses import NormalizedMeanSquaredError

sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Reservoirs")
from ESNPCA import ESNPCA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(f"D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Data/3BP_0.csv")
data = df[['x','y']].values
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = torch.tensor(data).to(device).float()

data = data[::50]
io_size=2
# Define training setup
# criterion
n_input=2000
warmup=200

model=ESNPCA(io_size, 4096, io_size, components=10, warmup=warmup, spectral_radius=1, sparsity=0, leaking_rate=0.5).to(device)

n_init=100
input = data[n_init:n_init+n_input]
model.fit(input)

n_init=3000
input = data[n_init:n_init+n_input]
target = input[warmup + 1:]

pred = model.predict(input)
criterion = NormalizedMeanSquaredError
loss = criterion(pred.unsqueeze(0), target.unsqueeze(0))
print(f"Loss: {loss.item():.4f}")

for v in range(io_size):
    plt.subplot(1,io_size,v+1)
    plt.plot(range(n_input), input[:,v].cpu().detach(), label="True (with warmup)")
    plt.plot(range(1+warmup, n_input), pred[:,v].cpu().detach(), label="Predicted (no warmup)")
    plt.axvline(x=warmup, linestyle="--", color="gray")
plt.legend()
plt.show()

## GENERATION

