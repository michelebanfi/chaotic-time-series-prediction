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
input_len = 10

# Define the model parameters
io_size = 2
num_epochs = 2

### LOAD DATA
# Load the data
print("Loading data...")
train_t, train_dataloader, val_t, val_dataloader = loadData(pred_len, input_len, file="3BP", train_samples=10, val_samples=10, sampling_rate=10)
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
scheduler = StepLR(optimizer, step_size=4, gamma=0.45)
print("Reservoir training...")
# start counting the time
start = time.time()
# Train the model
val_results, train_losses = (
    evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_dataloader, device, scheduler))
# save model
if diego:
    torch.save(val_results['model'].state_dict(), "D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Models/best_model.pth")
else:
    torch.save(val_results['model'].state_dict(), "Models/best_model.pth")
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
scheduler = StepLR(optimizer, step_size=4, gamma=0.45)
# start counting the time
start = time.time()
# Train the benchmark model
val_results_benchmark, train_losses_benchmark = (
    evaluate(num_epochs, criterion, optimizer, modelBenchmark, train_dataloader, val_dataloader, device, scheduler))
# save model
if diego:
    torch.save(val_results_benchmark['model'].state_dict(), "D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Models/best_model_benchmark.pth")
else:
    torch.save(val_results_benchmark['model'].state_dict(), "Models/best_model_benchmark.pth")
# stop counting the time
end = time.time()
print('Time elapsed: ', end - start, "s")
print(30*"-")


print("Plot validation...")
# Plotting the predictions
plt.figure(figsize=(15, 15))

how_many_plots = min(4, len(val_dataloader.dataset))
n_sequences = len(val_dataloader.dataset)
sequence_to_plot = torch.randint(0, n_sequences, (how_many_plots,))

batch_size = val_results['targets'][0].size(0)
batch_to_plot = torch.randint(0, batch_size, (how_many_plots,))

for plot in range(how_many_plots):
    seq = sequence_to_plot[plot].item()
    batch = batch_to_plot[plot].item()
    for var in range(io_size):
    # Plotting the predictions
        plt.subplot(how_many_plots, io_size, io_size*plot + var + 1)
        plt.plot(val_results['inputs'][seq][batch,:,var].cpu(), label='Input')
        plt.plot(val_results['targets'][seq][batch,:,var].cpu(), label='Target')
        plt.plot(val_results['predictions'][seq][batch,:,var].cpu(), label='Predicted (Reservoir)')
        plt.plot(val_results_benchmark['predictions'][seq][batch,:,var].cpu(), label='Predicted (Benchmark)')
        plt.xlabel('Time step')
        plt.legend()
        plt.grid()

plt.tight_layout()
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Media/r3bp_prediction.png')
else:
    plt.savefig('Media/r3bp_prediction.png')
plt.close()




print("Generating...")
# use the trained models to predict the validation data and compute the NRMSE
# firstly load the data
if diego:
    df = pd.read_csv("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Data/3BP_test.csv")
else:
    df = pd.read_csv("Data/3BP_test.csv")
data = torch.tensor(df[['x', 'y']].values).float()
t = df['time'].values

# split the data into warmup and test
starting_point = torch.randint(0,data.size(0), (1,))
n_input = 100
n_pred = 50
data_train, data_test = data[:n_input], data[n_input:n_input+n_pred]
t_train, t_test = t[:n_input], t[n_input:n_input+n_pred]

# unsqueeze the data
data_train = data_train.unsqueeze(0)

# evaluate the models
model.pred_len = data_test.size(0)
modelBenchmark.pred_len = data_test.size(0)

outputs = model(data_train.to(device))
outputs_benchmark = modelBenchmark(data_train.to(device))

# plot the 3 variables separated
plt.figure(figsize=(15, 15))
for i in range(io_size):
    plt.subplot(io_size, 1, i+1)
    plt.plot(t_train, data_train[0, :, i].cpu(), label='Train')
    plt.plot(t_test, data_test[:, i].cpu(), label='Test')
    plt.plot(t_test, outputs[0, :, i].cpu().detach(), label='Predicted (Reservoir)')
    plt.plot(t_test, outputs_benchmark[0, :, i].cpu().detach(), label='Predicted (Benchmark)')
    plt.xlabel('Time')
    plt.ylabel(f'Variable {i+1}')
    plt.legend()
    plt.grid()
plt.tight_layout()
if diego:
    plt.savefig('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/RestrictedThreeBodyProblem/Media/r3bp_generations.png')
else:
    plt.savefig('Media/lorenz_generations.png')
plt.close()

