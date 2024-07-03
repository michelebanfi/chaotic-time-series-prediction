import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
diego = True
if diego:
    sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
    from DataEvaluator import evaluate
    from DataLoader import loadData
    from Losses import NormalizedMeanSquaredError

    sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Benchmarks")
    from GRU import GRU
    from LSTM import LSTM

    sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Reservoirs")
    from GRUReservoir import GRUReservoir
    from LSTMReservoir import LSTMReservoir
    from ESNReservoir import ESNReservoir
else:
    from Utils.DataEvaluator import  evaluate
    from Utils.DataLoader import loadData
    from Benchmarks.GRU import GRU
    from Reservoirs.ESNReservoir import ESNReservoir
    from Reservoirs.GRUReservoir import GRUReservoir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# take the data from the .csv file
df = pd.read_csv("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Data/3BP_0.csv")
df = df[['x', 'y']]

in_len = 1
out_len = 50
starting_point = torch.randint(0, df.shape[0]-out_len, size=(1,)).item()
warmup = df[starting_point:starting_point+in_len]
input = torch.tensor(warmup.values).float().unsqueeze(0).to(device)

results = []
for i in range(5):
    model = ESNReservoir(2, 4096, 2, pred_len=out_len).to(device)
    model.eval()
    result = model(input)
    results.append(result)

# plot the result in 2 separate plots one for x and y
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.ylim(-0.1,0.1)
plt.title("x")
for result in results:
    plt.plot(result[0, :, 0].cpu().detach().numpy())
plt.grid()

plt.subplot(2, 1, 2)
plt.ylim(-0.1,0.1)
plt.title("y")
for result in results:
    plt.plot(result[0, :, 1].cpu().detach().numpy())
plt.grid()

plt.show()
