import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

import sys
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

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on:", device)
print(30*"-")

# Define sequences length
pred_len = 1
input_len = 100

# Define the model parameters
io_size = 3
num_epochs = 100

### LOAD DATA
# Load the data
print("Loading data...")
train_t, train_dataloader, val_t, val_dataloader = loadData(pred_len, input_len, file="lorenz", train_samples=200, val_samples=20)
print("Train batches:", len(train_dataloader))
print("Train input sequences:", len(train_dataloader.dataset))
print("Validation batches:", len(val_dataloader))
print("Validation input sequences:", len(val_dataloader.dataset))
print(30*"-")


# init the models
model = ESNReservoir(io_size, 1024, io_size, pred_len=pred_len).to(device)
modelBenchmark = GRU(io_size, 512, io_size, pred_len=pred_len, num_layers=1).to(device)






### RESERVOIR
# Define training setup
# criterion
criterion = NormalizedMeanSquaredError
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
print("Reservoir training...")
# start counting the time
start = time.time()
# Train the model
val_results, train_losses = (
    evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_dataloader, device, scheduler))
# stop counting the time
end = time.time()
print('Time elapsed: ', end - start, "s")
print(30*"-")





### BENCHMARK MODEL
print("Benchmark training...")
# training setup
# criterion
criterion = NormalizedMeanSquaredError
# optimizer
optimizer = optim.Adam(modelBenchmark.parameters(), lr=0.001)
# scheduler
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
# start counting the time
start = time.time()
# Train the benchmark model
val_results_benchmark, train_losses_benchmark = (
    evaluate(num_epochs, criterion, optimizer, modelBenchmark, train_dataloader, val_dataloader, device, scheduler))
# stop counting the time
end = time.time()
print('Time elapsed: ', end - start, "s")
print(30*"-")



# Plotting the predictions
plt.figure(figsize=(15, 15))

how_many_plots = min(4, len(val_dataloader.dataset))
n_sequences = len(val_dataloader.dataset)
sequence_to_plot = torch.randint(0, n_sequences, (how_many_plots,))

batch_size = val_results['targets'][0].size(0)
batch_to_plot = torch.randint(0, batch_size, (how_many_plots,))

for plot in range(how_many_plots - how_many_plots%2):
    seq = sequence_to_plot[plot].item()
    batch = batch_to_plot[plot].item()
    for var in range(io_size):
    # Plotting the predictions
        plt.subplot(how_many_plots // 2, 3, plot + 1)
        plt.plot(val_results['inputs'][seq][batch,:,var].cpu(), label='Input')
        plt.plot(val_results['targets'][seq][batch,:,var].cpu(), label='Target')
        plt.plot(val_results['predictions'][seq][batch,:,var].cpu(), label='Predicted (Reservoir)')
        plt.plot(val_results_benchmark['predictions'][seq][batch,:,var].cpu(), label='Predicted (Benchmark)')
        plt.xlabel('Time step')
        plt.legend()
        plt.grid()

plt.tight_layout()
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Media/lorenz_prediction.png')
else:
    plt.savefig('Media/lorenz_prediction.png')
plt.close()
